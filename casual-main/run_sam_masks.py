
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_color_masks import build_color_masks


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SAM')
    parser.add_argument(
        '--dataset', default='causal-lerf',
        help='lerf-ovs、Ref-lerf,causal-lerf'
    )
    parser.add_argument(
        '--scene_name', default='ramen',
        help='saved_masks figurines、ramen）'
    )
    cli_args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    base_dir = script_dir
    dataset    = cli_args.dataset
    scene_name = cli_args.scene_name
    data_path  = base_dir / "data" / dataset / scene_name
    save_dir   = repo_root / "saved_masks" / dataset / scene_name

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    level     = '0' if 'scannet' in dataset else '2'
    snpy_level = 0   if 'scannet' in dataset else 3

    export_script = base_dir / 'export_3d_to_mask_mapping.py'
    mapping_file = save_dir / '3d_object_to_masks_mapping.json'
    if not export_script.is_file():
        print(f"[run_sam_masks] missing {export_script}")
        return False

    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(export_script),
            '--save_dir', str(save_dir),
            '--data_path', str(data_path),
            '--output', str(mapping_file),
            '--level', level,
        ],
        cwd=str(base_dir),
        check=True,
    )

    try:
        build_color_masks(
            mapping_file=str(mapping_file),
            language_features_dir=str(data_path / 'language_features_test'),
            images_dir=str(data_path / 'images'),
            output_dir=str(save_dir / 'color_masks'),
            snpy_level=snpy_level,
        )
    except Exception:
        return False
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
