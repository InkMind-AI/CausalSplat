
import os
import re
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def build_color_masks(
    mapping_file='',
    language_features_dir='',
    images_dir='',
    output_dir='',
    snpy_level=3
):

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    total_objects = mapping['total_3d_objects']
    objects = mapping['objects']

    output_path = Path(output_dir)
    images_path = Path(images_dir)
    lang_path   = Path(language_features_dir)

    npy_cache   = {}
    image_cache = {}

    def load_snpy(view_name):
        if view_name not in npy_cache:
            npy_path = lang_path / f"{view_name}_s.npy"
            if not npy_path.exists():
                npy_cache[view_name] = None
            else:
                npy_cache[view_name] = np.load(str(npy_path))
        return npy_cache[view_name]

    _is_scannet = "scannet" in str(images_path).lower()
    _actual_images_path = images_path.parent / "color" if _is_scannet else images_path

    def load_image(frame_name):
        if frame_name not in image_cache:
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                img_path = _actual_images_path / f"{frame_name}{ext}"
                if img_path.exists():
                    image_cache[frame_name] = cv2.imread(str(img_path))
                    break
            else:
                image_cache[frame_name] = None
        return image_cache[frame_name]
        
    for obj in tqdm(objects, desc=""):
        obj_id     = obj['object_id']
        masks_info = obj['masks']

        if not masks_info:
            continue

        obj_out_dir = output_path / f"object_{obj_id:04d}"
        obj_out_dir.mkdir(parents=True, exist_ok=True)

        for mask_info in masks_info:
            global_index = mask_info['global_index']
            view_name    = mask_info['view_name']  
            mask_index   = mask_info['mask_index']

            seg_maps = load_snpy(view_name)
            if seg_maps is None:
                continue
            seg_map = seg_maps[snpy_level]  

            snpy_mask = (seg_map == mask_index).astype(np.uint8) * 255  

            frame_name = view_name  
            image = load_image(frame_name)
            if image is None:

                h, w = snpy_mask.shape[:2]
                final_mask = np.full((h, w, 3), 255, dtype=np.uint8)
                mask_bool = snpy_mask > 0  
            else:
                if snpy_mask.shape[:2] != image.shape[:2]:
                    snpy_mask_resized = cv2.resize(
                        snpy_mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    snpy_mask_resized = snpy_mask

                image_valid = (np.max(image, axis=2) > 0).astype(np.uint8) * 255  

                mask_bool = (snpy_mask_resized > 0) & (image_valid > 0)

                color_out = image.copy()
                color_out[~mask_bool] = 255  
                final_mask = color_out

            out_file = obj_out_dir / f"global_{global_index:05d}_{view_name}_mask{mask_index}.png"
            cv2.imwrite(str(out_file), final_mask)
