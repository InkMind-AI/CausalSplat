
import re
from pathlib import Path

import cv2
import numpy as np

_IMAGE_NAME = re.compile(r'^global_\d+_(.+)_mask(\d+)\.[^.]+$')


def crop_object_image(image_path, pad=5, min_width=128):
    
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read object image: {path}')
    height, width = img.shape[:2]
    info = {'source': str(path), 'original_size': [width, height],
            'bbox_xyxy': [0, 0, width, height], 'mask_source': None,
            'status': 'no_mask_full_view', 'foreground_pixels': None}
    match = _IMAGE_NAME.match(path.name)
    mask = None
    if match and path.parent.parent.name == 'color_masks':
        scene_root = path.parent.parent.parent
        dataset = scene_root.parent.name
        repo = scene_root.parent.parent.parent
        mask_path = repo / 'casual-main' / 'data' / dataset / scene_root.name / 'language_features_test' / (match[1] + '_s.npy')
        if mask_path.exists():
            maps = np.load(str(mask_path), mmap_mode='r', allow_pickle=False)
            mask_id = int(match[2])
            mask = maps == mask_id
            if mask.ndim == 3:
                mask = np.any(mask, axis=0)
            if mask.ndim != 2:
                raise ValueError(f'Unexpected SAM array shape {maps.shape}: {mask_path}')
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            info['mask_source'] = str(mask_path)
            info['foreground_pixels'] = int(mask.sum())
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        x0, x1 = max(0, int(xs.min())-pad), min(width, int(xs.max())+pad+1)
        y0, y1 = max(0, int(ys.min())-pad), min(height, int(ys.max())+pad+1)
        img = img[y0:y1, x0:x1]
        info.update(bbox_xyxy=[x0,y0,x1,y1], status='sam_mask_crop')
    elif mask is not None:
        info['status'] = 'empty_mask_full_view'
    info['crop_size_before_resize'] = [img.shape[1], img.shape[0]]
    if img.shape[1] < min_width:
        scale = min_width / img.shape[1]
        img = cv2.resize(img, (min_width, max(1, round(img.shape[0]*scale))))
    info['output_size'] = [img.shape[1], img.shape[0]]
    return img, info


def encode_object_image(image_path, extension='.jpg'):
    crop, info = crop_object_image(image_path)
    options = [cv2.IMWRITE_JPEG_QUALITY, 85] if extension in ('.jpg','.jpeg') else []
    ok, buffer = cv2.imencode(extension, crop, options)
    if not ok:
        raise ValueError(f'Cannot encode cropped object image: {image_path}')
    return buffer.tobytes(), info
