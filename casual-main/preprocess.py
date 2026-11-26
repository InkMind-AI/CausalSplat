import os
import sys
import random
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_LANGSPLAT_ROOT = os.path.join(CURRENT_DIR, 'third_party', 'segment-anything-langsplat')
if SAM_LANGSPLAT_ROOT not in sys.path:
    sys.path.insert(0, SAM_LANGSPLAT_ROOT)

import numpy as np
import torch
from tqdm import tqdm
from segment_anything_langsplat import SamAutomaticMaskGenerator, sam_model_registry

import cv2


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate SAM segmentation maps"

    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:]), dtype=torch.int32)
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Generating SAM masks", leave=False):
        try:
            seg_map = extract_sam_seg_maps(img.unsqueeze(0), sam_encoder)
        except Exception:
            raise ValueError(i + 1)

        seg_map_tensor = []
        lengths = []
        for mode in ['default', 's', 'm', 'l']:
            curr_map = seg_map.get(mode)
            if curr_map is None:
                curr_map = -np.ones(img.shape[1:], dtype=np.int32)
                lengths.append(0)
            else:
                lengths.append(int(curr_map.max() + 1) if curr_map.max() >= 0 else 0)
            seg_map_tensor.append(torch.from_numpy(curr_map))

        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths_cumsum)):
            lengths_cumsum[j] += lengths_cumsum[j - 1]

        for j in range(1, len(seg_map_tensor)):
            curr = seg_map_tensor[j]
            if lengths[j] == 0:
                continue
            curr[curr != -1] += lengths_cumsum[j - 1]

        seg_maps[i] = torch.stack(seg_map_tensor, dim=0)

    mask_generator.predictor.model.to('cpu')

    for i in range(seg_maps.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        sava_numpy(save_path, seg_maps[i])


def sava_numpy(save_path, seg_maps):
    save_path_s = save_path + '_s.npy'
    np.save(save_path_s, seg_maps.numpy())


def extract_sam_seg_maps(image, sam_encoder):
    return sam_encoder(torch.cat([image]))


def get_seg_img(mask, image):
    image = image.copy()
    
    image[mask['segmentation']==0] = np.array([255, 255, 255], dtype=np.uint8)

    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y + h, x:x + w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou

            
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    
    masks_new = ()
    for masks_lvl in args:
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    masks_default, masks_s, masks_m, masks_l = masks_update(
        masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5
    )


    def mask2segmap(masks, image):
      
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            seg_map[masks[i]['segmentation']] = i
        return seg_map

    seg_maps = {}  
    seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_maps['l'] = mask2segmap(masks_l, image)
    return seg_maps


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                 
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'language_features_test')

    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)
