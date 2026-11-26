#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build_mapping(labels: np.ndarray, idx_map: list[dict], level: int) -> dict:
    if len(labels) != len(idx_map):
        raise SystemExit(
            f"[export_3d_to_mask_mapping] level {level}: {len(labels)} labels but "
            f"{len(idx_map)} index entries; re-run cluster_objects.py"
        )
    object_ids = sorted(int(i) for i in np.unique(labels) if i != -1)
    if object_ids != list(range(len(object_ids))):
        raise SystemExit(f"[export_3d_to_mask_mapping] level {level}: non-contiguous object ids {object_ids}")

    masks_by_object: dict[int, list[dict]] = {i: [] for i in object_ids}
    for global_index, (entry, label) in enumerate(zip(idx_map, labels)):
        if label == -1:  
            continue
        masks_by_object[int(label)].append(
            {
                "global_index": global_index,
                "view_name": entry["view_name"],
                "mask_index": int(entry["mask_id"]),
            }
        )
    return {
        "level": level,
        "total_3d_objects": len(object_ids),
        "objects": [{"object_id": i, "masks": masks_by_object[i]} for i in object_ids],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export 3D object -> 2D mask mapping")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--data_path", default="", help="scene dir; if given, check the SAM id maps exist")
    p.add_argument("--output", required=True)
    p.add_argument("--level", type=int, default=2)
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    labels_path = save_dir / "multi_lvl_cluster_labels.pth"
    idx_path = save_dir / "multi_lvl_global_idx_map.json"
    for path in (labels_path, idx_path):
        if not path.is_file():
            raise SystemExit(f"[export_3d_to_mask_mapping] missing {path}; run cluster_objects.py first")

    all_labels = torch.load(labels_path, weights_only=False)  
    with open(idx_path, encoding="utf-8") as f:
        all_idx_map = json.load(f)
    if not 0 <= args.level < len(all_labels):
        raise SystemExit(f"[export_3d_to_mask_mapping] level {args.level} not in 0..{len(all_labels) - 1}")

    mapping = build_mapping(np.asarray(all_labels[args.level]), all_idx_map[args.level], args.level)

    if args.data_path:
        lang_dir = Path(args.data_path) / "language_features_test"
        views = {m["view_name"] for obj in mapping["objects"] for m in obj["masks"]}
        missing = sorted(v for v in views if not (lang_dir / f"{v}_s.npy").is_file())
        if missing:
            raise SystemExit(f"[export_3d_to_mask_mapping] {len(missing)} id maps missing in {lang_dir}, e.g. {missing[:3]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    n_masks = sum(len(o["masks"]) for o in mapping["objects"])
    print(f"[export_3d_to_mask_mapping] level={args.level} objects={mapping['total_3d_objects']} masks={n_masks} -> {output}")


if __name__ == "__main__":
    main()
