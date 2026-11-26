#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_gs_python() -> str:
    for p in (
        os.environ.get("PY"),
        os.path.expanduser("~/miniconda3/envs/causalsplat/bin/python"),
        os.path.expanduser("~/venvs/zerosplat/bin/python"),
    ):
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return sys.executable


DEFAULT_GS_PYTHON = _default_gs_python()


def _run(name: str, cmd: list[str], *, cwd: str, env: dict | None = None) -> None:
    print(f"[Pipeline] {name}")
    print(f"[Pipeline] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    if ret.returncode != 0:
        raise SystemExit(f"[Pipeline] {name} failed (exit={ret.returncode})")
    print(f"\n[Pipeline] {name} finished.\n")


def scene_paths(base_data_dir: str, dataset: str, scene_name: str) -> tuple[str, str]:
    base = os.path.abspath(base_data_dir)
    source_path = os.path.join(base, dataset, scene_name)
    model_path = os.path.join(source_path, "gs")
    return source_path, model_path


def validate_scene_root(source_path: str) -> None:
    if not os.path.isdir(source_path):
        raise SystemExit(f"[Pipeline] scene directory not found: {source_path}")
    has_colmap = os.path.isdir(os.path.join(source_path, "sparse"))
    has_nerf = os.path.isfile(os.path.join(source_path, "transforms_train.json"))
    if not has_colmap and not has_nerf:
        raise SystemExit(
            "[Pipeline] scene is missing COLMAP (sparse/) and NeRF "
            f"(transforms_train.json): {source_path}"
        )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="CausalSplat pipeline launcher (train 3D Gaussian Splatting scene)"
    )
    p.add_argument(
        "--dataset",
        "--dataset_type",
        dest="dataset",
        type=str,
        default="causal-lerf",
        help="Dataset folder under base_data_dir (e.g. causal-lerf, causal-scannet)",
    )
    p.add_argument(
        "--scene_name",
        "--scene",
        "--dataset_name",
        dest="scene_name",
        type=str,
        default="ramen",
        help="Scene folder name (e.g. ramen, scene0000_00)",
    )
    p.add_argument(
        "--base_data_dir",
        type=str,
        default=os.path.join(ROOT, "data"),
        help="Root containing <dataset>/<scene_name>/",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Training iterations (default: 30000)",
    )
    p.add_argument(
        "--gs_python",
        type=str,
        default="",
        help=f"Python for train_scene.py (default: {DEFAULT_GS_PYTHON} or current)",
    )
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Print resolved paths only; do not run train_scene.py",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Passed through to train_scene.py",
    )
    return p.parse_known_args()


def main() -> None:
    args, extra_train_args = parse_args()

    dataset = args.dataset
    scene_name = args.scene_name
    source_path, model_path = scene_paths(args.base_data_dir, dataset, scene_name)

    if args.gs_python:
        gs_py = os.path.abspath(args.gs_python)
        if not os.path.isfile(gs_py):
            raise SystemExit(f"[Pipeline] gs_python not found: {gs_py}")
    elif os.path.isfile(DEFAULT_GS_PYTHON):
        gs_py = DEFAULT_GS_PYTHON
    else:
        gs_py = sys.executable

    iterations = args.iterations if args.iterations is not None else 30_000

    print(f"[Pipeline] dataset={dataset}")
    print(f"[Pipeline] scene_name={scene_name}")
    print(f"[Pipeline] base_data_dir={os.path.abspath(args.base_data_dir)}")
    print(f"[Pipeline] source_path={source_path}")
    print(f"[Pipeline] model_path={model_path}")
    print(f"[Pipeline] iterations={iterations}")
    print(f"[Pipeline] gs_python={gs_py}")

    validate_scene_root(source_path)

    if args.skip_train:
        print("[Pipeline] --skip_train: not running train_scene.py")
        return

    train_script = os.path.join(ROOT, "train_scene.py")
    cmd = [
        gs_py,
        train_script,
        "-s",
        source_path,
        "-m",
        model_path,
        "--iterations",
        str(iterations),
    ]
    if args.quiet:
        cmd.append("--quiet")
    cmd.extend(extra_train_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    _run("train_scene.py", cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
