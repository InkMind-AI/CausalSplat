import numpy as np
import torch

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def normalize(x: np.ndarray) -> np.ndarray:
    
    return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def pad_poses(p: np.ndarray) -> np.ndarray:
    
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
    
    return p[..., :3, :4]

def integrate_weights(w):
    
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    
    cw0 = np.concatenate([np.zeros(shape), cw, np.ones(shape)], axis=-1)
    return cw0

def interp(x, xp, fp):
    
    x_flat = x.reshape(-1, x.shape[-1])
    xp_flat = xp.reshape(-1, xp.shape[-1])
    fp_flat = fp.reshape(-1, fp.shape[-1])

    
    ret_flat = np.array([np.interp(xf, xpf, fpf) for xf, xpf, fpf in zip(x_flat, xp_flat, fp_flat)])

    
    ret = ret_flat.reshape(x.shape)
    return ret

def sorted_interp(x, xp, fp):
    
    
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x):
        
        
        x0 = np.max(np.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = np.min(np.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    fp0, fp1 = find_interval(fp)
    xp0, xp1 = find_interval(xp)
    with np.errstate(divide='ignore', invalid='ignore'):
        offset = np.clip(np.nan_to_num((x - xp0) / (xp1 - xp0), nan=0.0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret

def invert_cdf(u, t, w_logits, use_gpu_resampling=False):
    
    
    from scipy.special import softmax
    w = softmax(w_logits, axis=-1)
    cw = integrate_weights(w)

    
    interp_fn = interp if use_gpu_resampling else sorted_interp  
    t_new = interp_fn(u, cw, t)
    return t_new

def sample(rng,
           t,
           w_logits,
           num_samples,
           single_jitter=False,
           deterministic_center=False,
           use_gpu_resampling=False):
    
    eps = np.finfo(np.float32).eps

    
    if rng is None:
        
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = np.linspace(pad, 1. - pad - eps, num_samples)
        else:
            u = np.linspace(0, 1. - eps, num_samples)
            u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = (
            np.linspace(0, 1 - u_max, num_samples) +
            rng.uniform(size=t.shape[:-1] + (d,), high=max_jitter))

    return invert_cdf(u, t, w_logits, use_gpu_resampling=use_gpu_resampling)

def transform_poses_pca(poses: np.ndarray):
    
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    

    return poses_recentered, transform, scale_factor

def generate_ellipse_path_from_poses(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    
    
    center = focus_point_fn(poses)
    
    offset = np.array([center[0], center[1], 0])

    
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 80, axis=0)
    

    
    low = -sc + offset
    high = sc + offset
    
    z_low = np.percentile((poses[:, :3, 3]), 0, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 100, axis=0)

    def get_positions(theta):
        
        
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                        (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    print('theta[0]', theta[0])

    if const_speed:
        
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    
    positions = positions[:-1]

    
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def invert_transform_poses_pca(poses_recentered, transform, scale_factor):
    poses_recentered[:, :3, 3] /= scale_factor
    transform_inv = np.linalg.inv(transform)
    poses_original = unpad_poses(transform_inv @ pad_poses(poses_recentered))
    return poses_original

def generate_ellipse_path_from_camera_infos(
        cameras: None,
        n_frames: int = 120,
        const_speed: bool = False,
        z_variation: float = 0.,
        z_phase: float = 0.
    ):
    
    poses = np.array([np.linalg.inv(getWorld2View2(cam.R, cam.T))[:3, :4] for cam in cameras])
    poses[:, :, 1:3] *= -1
    poses, transform, scale_factor = transform_poses_pca(poses)
    render_poses = generate_ellipse_path_from_poses(poses, n_frames, const_speed, z_variation, z_phase)
    render_poses = invert_transform_poses_pca(render_poses, transform, scale_factor)
    render_poses[:, :, 1:3] *= -1

    def getProjectionMatrix(znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    fov = 50
    import math
    render_height = 738
    render_width = 994
    aspect_ratio = render_width / render_height
    fovX = 2 * math.atan(math.tan(math.radians(fov) / 2) * aspect_ratio)
    fovY = 2 * math.atan(math.tan(math.radians(fov) / 2))

    ret = []
    for pose in render_poses:
        R = pose[:3, :3]
        c2w = np.eye(4)
        c2w[:3, :4] = pose
        T = np.linalg.inv(c2w)[:3, 3]

        
        
        
        from copy import deepcopy
        tmp_camera = deepcopy(cameras[0])

        tmp_camera.R = R
        tmp_camera.T = T

        tmp_camera.FoVx = fovX
        tmp_camera.FoVy = fovY

        tmp_camera.image_name = 'lerf_eval_camera0'

        tmp_camera.image_width = render_width
        tmp_camera.image_height = render_height


        tmp_camera.world_view_transform = torch.tensor(getWorld2View2(tmp_camera.R, tmp_camera.T)).transpose(0, 1).cuda()
        tmp_camera.projection_matrix = getProjectionMatrix(znear=tmp_camera.znear, zfar=tmp_camera.zfar, fovX=tmp_camera.FoVx, fovY=tmp_camera.FoVy).transpose(0,1).cuda()
        tmp_camera.full_proj_transform = (tmp_camera.world_view_transform.unsqueeze(0).bmm(tmp_camera.projection_matrix.unsqueeze(0))).squeeze(0)
        tmp_camera.camera_center = tmp_camera.world_view_transform.inverse()[3, :3]
        ret.append(tmp_camera)
    return ret
    

def get_spherial_video_trace(cameras):
    import math
    def getProjectionMatrix(znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    from copy import deepcopy
    def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    trans = np.array([
        [
            1.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0
        ]
    ])
    scale = 1.0 

    def trans_back_c2w(
        c2w, trans = trans, scale = scale, camera_convention = 'opencv',
    ):
        c2w = deepcopy(c2w)
        c2w[:3, 3] /= scale
        inv_transform = np.linalg.inv(
            np.concatenate(
                (
                    trans,
                    np.array([[0, 0, 0, 1]]),
                ),
                0,
            )
        )
        output_c2w = inv_transform @ c2w

        if camera_convention == "opencv":
            output_c2w[0:3, 1:3] *= -1
        elif camera_convention == "opengl":
            pass
        else:
            raise ValueError(f"Camera convention {camera_convention} is not supported.")
        return output_c2w


    c2ws = []
    for cam in cameras:
        
        c2ws.append(cam.world_view_transform.transpose(0, 1).inverse())
    c2ws = torch.stack(c2ws, dim = 0)

    poses_ = c2ws.detach().cpu().numpy().copy()

    
    rot_phi = lambda phi : np.array([ 
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).astype(np.float32)

    rot_theta = lambda th : np.array([ 
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).astype(np.float32)

    
    rot_gamma = lambda ga : np.array([ 
        [np.cos(ga),-np.sin(ga),0,0],
        [np.sin(ga), np.cos(ga),0,0],
        [0,0,1,0],
        [0,0,0,1]]).astype(np.float32)

    def pose_spherical(gamma, phi, t):
        c2w = np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]]).astype(np.float32)
        
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_gamma(gamma/180.*np.pi) @ c2w
        c2w[:3, 3] = t
        return c2w


    movie_render_kwargs=dict(
            shift_x=0.0,  
            shift_y=0.0, 
            shift_z=0.0,
            scale_r=0.0,
            pitch_deg=55,
            
    )
    centroid = poses_[:,:3,3].mean(0)
    radcircle = movie_render_kwargs.get('scale_r', 0) * np.linalg.norm(poses_[:,:3,3] - centroid, axis=-1).mean()
    centroid[0] += movie_render_kwargs.get('shift_x', 0)
    centroid[1] += movie_render_kwargs.get('shift_y', 0)
    centroid[2] += movie_render_kwargs.get('shift_z', 0)
    up_rad = movie_render_kwargs.get('pitch_deg', 0)


    render_poses = []
    camera_o = np.zeros_like(centroid)
    num_render = 45
    for th in np.linspace(0., 360., num_render):
        camera_o[0] = centroid[0] + radcircle * np.cos(th/180.*np.pi)
        camera_o[1] = centroid[1] + radcircle * np.sin(th/180.*np.pi)
        camera_o[2] = centroid[2]
        render_poses.append(pose_spherical(th+90.0, up_rad, camera_o))
        
    render_poses = torch.from_numpy(np.stack(render_poses, axis=0))

    tmp_camera = deepcopy(cameras[0])

    fov = 50

    render_height = 738
    render_width = 994
    aspect_ratio = render_width / render_height
    fovX = 2 * math.atan(math.tan(math.radians(fov) / 2) * aspect_ratio)
    fovY = 2 * math.atan(math.tan(math.radians(fov) / 2))

    ret = []
    for pose in render_poses:
        p = pose.cpu().numpy()
        p = trans_back_c2w(p)
        W2C_in_gs = np.linalg.inv(p)
        tmp_camera = deepcopy(cameras[0])

        tmp_camera.R = W2C_in_gs[:3, :3]
        tmp_camera.T = W2C_in_gs[:3, 3]

        tmp_camera.FoVx = fovX
        tmp_camera.FoVy = fovY

        tmp_camera.image_name = 'lerf_eval_camera0'

        tmp_camera.image_width = render_width
        tmp_camera.image_height = render_height


        tmp_camera.world_view_transform = torch.tensor(getWorld2View2(tmp_camera.R, tmp_camera.T)).transpose(0, 1).cuda()
        tmp_camera.projection_matrix = getProjectionMatrix(znear=tmp_camera.znear, zfar=tmp_camera.zfar, fovX=tmp_camera.FoVx, fovY=tmp_camera.FoVy).transpose(0,1).cuda()
        tmp_camera.full_proj_transform = (tmp_camera.world_view_transform.unsqueeze(0).bmm(tmp_camera.projection_matrix.unsqueeze(0))).squeeze(0)
        tmp_camera.camera_center = tmp_camera.world_view_transform.inverse()[3, :3]
        ret.append(tmp_camera)
    return ret
