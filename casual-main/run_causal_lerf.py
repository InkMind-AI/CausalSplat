#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA_ROOT = Path(os.environ.get("CAUSALSPLAT_DATA_ROOT", ROOT / "data"))
DEFAULT_LERF_OVS = DATA_ROOT / "lerf_ovs"
DEFAULT_CAUSAL_LERF_GT = DATA_ROOT / "causal-lerf"
SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def _require_under_data_root(path: Path, what: str) -> Path:
    resolved = path.resolve()
    root = DATA_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            f"[Pipeline] {what} must be under {DATA_ROOT}, refused {path}"
        ) from exc
    return resolved


def _default_python() -> str:
    for p in (
        os.environ.get("PY"),
        os.path.expanduser("~/miniconda3/envs/causalsplat/bin/python"),
        sys.executable,
    ):
        if p and Path(p).is_file():
            return str(Path(p).resolve())
    return sys.executable


def _run(name: str, cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    print(f"\n[Pipeline] {name}")
    print(f"[Pipeline] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(cwd), env=env)
    if ret.returncode != 0:
        raise SystemExit(f"[Pipeline] {name} failed (exit={ret.returncode})")
    print(f"[Pipeline] {name} finished.")


def _env_with_paths() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def scene_dir(base_dir: Path, scene: str) -> Path:
    return base_dir / "data" / "causal-lerf" / scene


def gs_dir(base_dir: Path, scene: str) -> Path:
    return scene_dir(base_dir, scene) / "gs"


def saved_dir(repo_root: Path, scene: str) -> Path:
    return repo_root / "saved_masks" / "causal-lerf" / scene


def gauss_ply_30k(base_dir: Path, scene: str) -> Path:
    
    
    return gs_dir(base_dir, scene) / "point_cloud" / "iteration_30000" / "scene_point_cloud.ply"


def feature_ply_path(base_dir: Path, scene: str, feature_iter: int) -> Path:
    
    return (
        gs_dir(base_dir, scene)
        / "point_cloud"
        / f"iteration_{feature_iter}"
        / "contrastive_feature_point_cloud.ply"
    )


def seg_scores_path(base_dir: Path, scene: str, feature_iter: int) -> Path:
    
    return (
        gs_dir(base_dir, scene)
        / "point_cloud"
        / f"iteration_{feature_iter}"
        / "multi_lvl_seg_scores.pth"
    )


def preprocess_done(sd: Path) -> bool:
    return (sd / "language_features_test").is_dir()


def color_masks_done(save: Path) -> bool:
    cm = save / "color_masks"
    return cm.is_dir() and any(cm.iterdir())


StageAction = Literal["run", "skip"]


def resolve_stage(
    stage: str,
    *,
    skip: bool,
    force: bool,
    artifact: Path,
    done: bool | None = None,
) -> StageAction:
    if skip and force:
        raise SystemExit(
            f"[Pipeline] {stage}: pass only one of --skip_{stage} or --force_{stage}"
        )
    if skip:
        return "skip"
    if force:
        return "run"
    is_done = done if done is not None else artifact.is_file() or artifact.is_dir()
    if is_done:
        raise SystemExit(
            f"[Pipeline] {stage}: output already exists ({artifact}). "
            f"Pass --skip_{stage} to skip or --force_{stage} to re-run."
        )
    return "run"


def ensure_cfg_args(base_dir: Path, scene: str) -> None:
    gsd = gs_dir(base_dir, scene)
    cfg = gsd / "cfg_args"
    if cfg.is_file():
        return
    gsd.mkdir(parents=True, exist_ok=True)
    sd = str(scene_dir(base_dir, scene).resolve())
    gd = str(gsd.resolve())
    cfg.write_text(
        "Namespace("
        f"sh_degree=3, feature_dim=32, init_from_3dgs_pcd=False, "
        f"source_path={sd!r}, model_path={gd!r}, feature_model_path='', "
        f"images='images', resolution=-1, white_background=False, "
        f"data_device='cuda', eval=False, need_masks=False, "
        f"allow_principle_point_shift=False)\n",
        encoding="utf-8",
    )
    print(f"[Pipeline] wrote {cfg}")


def causal_lerf_gt_ok(sd: Path) -> bool:
    test_json = sd / "json" / "test_json"
    return test_json.is_dir() and any(test_json.glob("*.json"))


def sync_causal_lerf_gt(
    base_dir: Path,
    scene: str,
    src_root: Path,
    *,
    force: bool = False,
) -> None:
    
    src = _require_under_data_root(src_root / scene, "causal-lerf GT")
    dst = scene_dir(base_dir, scene).resolve()
    if src == dst:
        print(f"[Pipeline] using dataset in place: {dst}")
        return
    if not src.is_dir():
        print(f"[Pipeline] Causal-LeRF GT source missing, skip restore: {src}")
        return
    if causal_lerf_gt_ok(dst) and not force:
        print(f"[Pipeline] Causal-LeRF GT ok: {dst / 'json' / 'test_json'}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("json", "mask"):
        src_item = src / name
        dst_item = dst / name
        if not src_item.exists():
            print(f"[Pipeline] Causal-LeRF GT missing {src_item}")
            continue
        if dst_item.exists() or dst_item.is_symlink():
            if dst_item.is_dir() and not dst_item.is_symlink():
                shutil.rmtree(dst_item)
            else:
                dst_item.unlink()
        shutil.copytree(src_item, dst_item, symlinks=True)
        print(f"[Pipeline] restored Causal-LeRF {name} -> {dst_item}")


def sync_from_lerf_ovs(base_dir: Path, scene: str, lerf_root: Path) -> None:
    src = _require_under_data_root(lerf_root / scene, "lerf_ovs scene")
    dst = scene_dir(base_dir, scene).resolve()
    if src == dst:
        print(f"[Pipeline] lerf_ovs source is the scene dir, skip copy: {dst}")
        return
    if not src.is_dir():
        raise SystemExit(f"[Pipeline] missing lerf_ovs scene: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    src_sparse = src / "sparse"
    dst_sparse = dst / "sparse"
    if src_sparse.is_dir() and not dst_sparse.is_dir():
        shutil.copytree(src_sparse, dst_sparse, symlinks=True)
        print(f"[Pipeline] synced sparse -> {dst_sparse}")

    

    lang_zip = src / "language_features.zip"
    lang_dir = dst / "language_features_test"
    if lang_zip.is_file() and not lang_dir.is_dir():
        lang_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(lang_zip) as zf:
            zf.extractall(lang_dir)
        print(f"[Pipeline] extracted language_features -> {lang_dir}")

    src_lang = src / "language_features"
    if src_lang.is_dir() and not lang_dir.is_dir():
        shutil.copytree(src_lang, lang_dir, symlinks=True)
        print(f"[Pipeline] synced language_features -> {lang_dir}")

    
    nested = lang_dir / "language_features"
    if lang_dir.is_dir() and nested.is_dir():
        for child in nested.iterdir():
            target = lang_dir / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        shutil.rmtree(nested, ignore_errors=True)
        print(f"[Pipeline] flattened nested language_features -> {lang_dir}")

    ensure_cfg_args(base_dir, scene)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CausalSplat on causal-lerf (dataset + scene_name shown at start)"
    )
    p.add_argument(
        "--dataset",
        default="causal-lerf",
        help="Dataset folder under base_dir/data (default: causal-lerf)",
    )
    p.add_argument(
        "--scene_name",
        "--scene",
        dest="scene_name",
        default="ramen",
        choices=SCENES,
        help="Scene name",
    )
    p.add_argument(
        "--base_dir",
        type=str,
        default=str(ROOT),
        help="casual-main root (contains data/)",
    )
    p.add_argument(
        "--repo_root",
        type=str,
        default=str(REPO),
        help="Causalsplat repo root (saved_masks/, run_pipeline.py)",
    )
    p.add_argument("--python", dest="py", default="", help="Python interpreter")
    p.add_argument(
        "--lerf_ovs_dir",
        type=str,
        default=str(DEFAULT_LERF_OVS),
        help="lerf_ovs folder under casual-main/data, used only with --sync_lerf_ovs",
    )
    p.add_argument(
        "--causal_lerf_gt_dir",
        type=str,
        default=str(DEFAULT_CAUSAL_LERF_GT),
        help="causal-lerf root under casual-main/data (json + mask per scene)",
    )
    p.add_argument(
        "--no_sync_causal_gt",
        action="store_true",
        help="Do not restore Causal-LeRF json/mask even if test_json is missing",
    )
    p.add_argument(
        "--force_sync_causal_gt",
        action="store_true",
        help="Overwrite existing json/mask with Causal-LeRF GT from --causal_lerf_gt_dir",
    )
    p.add_argument(
        "--sam_ckpt_path",
        type=str,
        default=os.environ.get("SAM_CKPT_PATH", ""),
        help="SAM ViT-H checkpoint for preprocess.py (or set SAM_CKPT_PATH)",
    )
    p.add_argument("--feature_iter", type=int, default=30000)
    p.add_argument("--sync_lerf_ovs", action="store_true", help="Copy sparse/gs/lang from lerf_ovs")
    p.add_argument("--skip_train", action="store_true", help="Skip train_scene")
    p.add_argument("--force_train", action="store_true", help="Re-run train_scene even if ply exists")
    p.add_argument("--skip_preprocess", action="store_true", help="Skip SAM preprocess")
    p.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Re-run preprocess even if language_features_test exists",
    )
    p.add_argument("--skip_affinity", action="store_true", help="Skip train_affinity_features")
    p.add_argument(
        "--force_affinity",
        action="store_true",
        help="Re-run affinity even if contrastive_feature_point_cloud.ply exists",
    )
    p.add_argument("--skip_cluster", action="store_true", help="Skip cluster_objects")
    p.add_argument(
        "--force_cluster",
        action="store_true",
        help="Re-run cluster_objects even if multi_lvl_seg_scores.pth exists",
    )
    p.add_argument("--skip_color_masks", action="store_true", help="Skip run_sam_masks")
    p.add_argument(
        "--force_color_masks",
        action="store_true",
        help="Re-run color_masks even if color_masks/ exists",
    )
    p.add_argument("--skip_vlm", action="store_true", help="Skip run_pipeline.py (VLM describe)")
    p.add_argument(
        "--force_vlm",
        action="store_true",
        help="Re-run VLM describe even if vlm_results.json exists",
    )
    p.add_argument("--skip_scene_graph", action="store_true", help="Skip build_cssg_scene_graph")
    p.add_argument(
        "--force_scene_graph",
        action="store_true",
        help="Re-run scene graph even if scene_graph_lvl2.json exists",
    )
    p.add_argument("--skip_reason", action="store_true", help="Skip reason_llm")
    p.add_argument(
        "--force_reason",
        action="store_true",
        help="Re-run reason_llm even if llm_reasoning_results exists",
    )
    p.add_argument(
        "--scene_graph_level",
        type=int,
        default=2,
        help="Level used for scene_graph / reason artifact paths (default: 2)",
    )
    p.add_argument("--wait_vlm", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print plan only")
    return p.parse_args()






def main() -> None:
    args = parse_args()
    _run_pipeline(args)


def _run_pipeline(args: argparse.Namespace) -> None:
    if args.dataset != "causal-lerf":
        print(f"[Pipeline] note: script tuned for causal-lerf; dataset={args.dataset}")

    base_dir = Path(args.base_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    scene = args.scene_name
    py = Path(args.py).resolve() if args.py else Path(_default_python())
    env = _env_with_paths()

    sd = scene_dir(base_dir, scene)
    print(f"[Pipeline] dataset={args.dataset}")
    print(f"[Pipeline] scene_name={scene}")
    print(f"[Pipeline] base_dir={base_dir}")
    print(f"[Pipeline] repo_root={repo_root}")
    print(f"[Pipeline] scene_path={sd}")
    print(f"[Pipeline] gs_path={gs_dir(base_dir, scene)}")
    print(f"[Pipeline] saved_masks={saved_dir(repo_root, scene)}")
    print(f"[Pipeline] python={py}")

    if not sd.is_dir():
        raise SystemExit(f"[Pipeline] scene not found: {sd}")

    if args.dataset == "causal-lerf" and not args.no_sync_causal_gt:
        sync_causal_lerf_gt(
            base_dir,
            scene,
            Path(args.causal_lerf_gt_dir),
            force=args.force_sync_causal_gt,
        )

    
    if args.sync_lerf_ovs:
        sync_from_lerf_ovs(base_dir, scene, Path(args.lerf_ovs_dir))

    ensure_cfg_args(base_dir, scene)

    save = saved_dir(repo_root, scene)
    lvl = args.scene_graph_level
    ply = gauss_ply_30k(base_dir, scene)
    lang_feat = sd / "language_features_test"
    cluster_labels = save / "multi_lvl_cluster_labels.pth"
    cluster_idx_map = save / "multi_lvl_global_idx_map.json"

    train_action = resolve_stage(
        "train", skip=args.skip_train, force=args.force_train, artifact=ply
    )
    preprocess_action = resolve_stage(
        "preprocess",
        skip=args.skip_preprocess,
        force=args.force_preprocess,
        artifact=lang_feat,
        done=preprocess_done(sd),
    )
    affinity_action = resolve_stage(
        "affinity",
        skip=args.skip_affinity,
        force=args.force_affinity,
        artifact=feature_ply_path(base_dir, scene, args.feature_iter),
    )
    cluster_action = resolve_stage(
        "cluster",
        skip=args.skip_cluster,
        force=args.force_cluster,
        artifact=seg_scores_path(base_dir, scene, args.feature_iter),
    )
    color_masks_action = resolve_stage(
        "color_masks",
        skip=args.skip_color_masks,
        force=args.force_color_masks,
        artifact=save / "color_masks",
        done=color_masks_done(save),
    )
    vlm_json = save / "vlm_results.json"
    vlm_action = resolve_stage(
        "vlm", skip=args.skip_vlm, force=args.force_vlm, artifact=vlm_json
    )
    sg_json = save / f"scene_graph_lvl{lvl}.json"
    scene_graph_action = resolve_stage(
        "scene_graph",
        skip=args.skip_scene_graph,
        force=args.force_scene_graph,
        artifact=sg_json,
    )
    reason_json = save / f"llm_reasoning_results_lvl{lvl}.json"
    reason_action = resolve_stage(
        "reason", skip=args.skip_reason, force=args.force_reason, artifact=reason_json
    )

    def _stage_label(name: str, action: StageAction) -> str:
        return f"{name} ({'skip' if action == 'skip' else 'run'})"

    plan: list[str] = []
    if args.sync_lerf_ovs:
        plan.append("sync_lerf_ovs (done)")
    plan.extend(
        [
            _stage_label("train_scene", train_action),
            _stage_label("preprocess", preprocess_action),
            _stage_label("train_affinity_features", affinity_action),
            _stage_label("cluster_objects", cluster_action),
            _stage_label("color_masks", color_masks_action),
            _stage_label("vlm_describe", vlm_action),
            _stage_label("build_cssg_scene_graph", scene_graph_action),
            _stage_label("reason_llm", reason_action),
        ]
    )
    print(f"[Pipeline] stages: {' -> '.join(plan)}")

    if args.dry_run:
        return

    

    sparse_ok = (sd / "sparse" / "0" / "images.bin").is_file()
    if not sparse_ok:
        raise SystemExit(f"[Pipeline] missing COLMAP sparse: {sd / 'sparse' / '0'}")

    if train_action == "run":
        _run(
            "main.py (train_scene)",
            [
                str(py),
                str(ROOT / "main.py"),
                "--dataset",
                args.dataset,
                "--scene_name",
                scene,
                "--base_data_dir",
                str(base_dir / "data"),
            ],
            cwd=ROOT,
            env=env,
        )
    else:
        print("[Pipeline] skip train (--skip_train)")

    if preprocess_action == "run":
        if not args.sam_ckpt_path or not Path(args.sam_ckpt_path).is_file():
            raise SystemExit(
                "[Pipeline] preprocess needs --sam_ckpt_path or SAM_CKPT_PATH "
                "(SAM vit_h .pth)"
            )
        _run(
            "preprocess.py",
            [
                str(py),
                str(ROOT / "preprocess.py"),
                "--dataset_path",
                str(sd),
                "--sam_ckpt_path",
                args.sam_ckpt_path,
                "--resolution",
                "-1",
            ],
            cwd=ROOT,
            env=env,
        )
    else:
        print("[Pipeline] skip preprocess (--skip_preprocess)")

    ply = gauss_ply_30k(base_dir, scene)
    if affinity_action == "run":
        if not ply.is_file():
            raise SystemExit(f"[Pipeline] affinity needs trained gaussians: {ply}")
        _run(
            "train_affinity_features.py",
            [
                str(py),
                str(ROOT / "train_affinity_features.py"),
                "-s",
                str(sd),
                "-m",
                str(gs_dir(base_dir, scene)),
                "--iteration",
                str(args.feature_iter),
            ],
            cwd=ROOT,
            env=env,
        )
    else:
        print("[Pipeline] skip affinity (--skip_affinity)")

    if cluster_action == "run":
        feature_ply = feature_ply_path(base_dir, scene, args.feature_iter)
        if not feature_ply.is_file():
            raise SystemExit(f"[Pipeline] cluster needs {feature_ply} (run affinity first)")
        if not lang_feat.is_dir():
            raise SystemExit(f"[Pipeline] cluster needs {lang_feat} (run preprocess first)")
        _run(
            "cluster_objects.py",
            [
                str(py),
                str(ROOT / "cluster_objects.py"),
                "--dataset",
                args.dataset,
                "--scene_name",
                scene,
                "--base_dir",
                str(base_dir),
                "--repo_root",
                str(repo_root),
                "--feature_iter",
                str(args.feature_iter),
            ],
            cwd=ROOT,
            env=env,
        )
    else:
        print("[Pipeline] skip cluster (--skip_cluster)")

    if color_masks_action == "run":
        for needed in (cluster_labels, cluster_idx_map):
            if not needed.is_file():
                raise SystemExit(
                    f"[Pipeline] color_masks needs {needed} "
                    "(run cluster_objects first, or pass --skip_color_masks)"
                )
        if not lang_feat.is_dir():
            raise SystemExit(
                f"[Pipeline] color_masks needs {lang_feat} "
                "(run preprocess or pass --skip_color_masks)"
            )
        _run(
            "run_sam_masks.py",
            [
                str(py),
                str(ROOT / "run_sam_masks.py"),
                "--dataset",
                args.dataset,
                "--scene_name",
                scene,
            ],
            cwd=ROOT,
            env=env,
        )
    else:
        print("[Pipeline] skip color_masks (--skip_color_masks)")

    if vlm_action == "run":
        vlm_cmd = [
            str(py),
            str(repo_root / "run_pipeline.py"),
            "--dataset",
            args.dataset,
            "--scene_name",
            scene,
        ]
        if args.wait_vlm:
            vlm_cmd.append("--wait_vlm")
        _run("run_pipeline.py (VLM)", vlm_cmd, cwd=str(repo_root), env=env)
    else:
        print("[Pipeline] skip vlm (--skip_vlm)")

    if scene_graph_action == "run":
        _run(
            "build_cssg_scene_graph.py",
            [
                str(py),
                str(ROOT / "build_cssg_scene_graph.py"),
                "--base_dir",
                str(base_dir),
                "--repo_root",
                str(repo_root),
                "--dataset",
                args.dataset,
                "--scene_name",
                scene,
                "--feature_iter",
                str(args.feature_iter),
            ],
            cwd=ROOT,
            env=env,
        )
    else:
        print("[Pipeline] skip scene_graph (--skip_scene_graph)")

    if reason_action == "run":
        reason_cmd = [
            str(py),
            str(ROOT / "reason_llm.py"),
            "--base_dir",
            str(base_dir),
            "--repo_root",
            str(repo_root),
            "--dataset",
            args.dataset,
            "--scene_name",
            scene,
        ]
        if args.wait_vlm:
            reason_cmd.append("--wait_vlm")
        _run("reason_llm.py", reason_cmd, cwd=ROOT, env=env)
    else:
        print("[Pipeline] skip reason (--skip_reason)")

    print("\n[Pipeline] All requested stages finished.")


if __name__ == "__main__":
    main()
