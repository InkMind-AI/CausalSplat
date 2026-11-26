#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import hdbscan
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams  
from gaussian_renderer import render_contrastive_feature  
from scene import FeatureGaussianModel, Scene  

FEATURE_DIM = 32

MASK_SIZE_THRESH = 400
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_EPS_PER_LEVEL = 0.1


def load_cfg_args(model_path: Path, scene_path: Path) -> tuple[Namespace, ModelParams, PipelineParams]:
    
    parser = ArgumentParser(description="cluster_objects")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    cmdline = parser.parse_args(["--model_path", str(model_path)])

    with open(model_path / "cfg_args") as f:
        cfg = eval(f.read(), {"Namespace": Namespace})  

    merged = vars(cfg).copy()
    for k, v in vars(cmdline).items():
        if v is not None:
            merged[k] = v
    
    merged["source_path"] = str(scene_path)
    merged["model_path"] = str(model_path)
    return Namespace(**merged), model, pipeline


@torch.no_grad()
def extract_mask_prototypes(scene, feature_gaussians, pipe, num_lvl: int, multi_lvl_dim: list[int]):
   
    prototypes: list[list[torch.Tensor]] = [[] for _ in range(num_lvl)]
    idx_map: list[list[dict]] = [[] for _ in range(num_lvl)]
    background = torch.zeros(FEATURE_DIM, dtype=torch.float32, device="cuda")

    for view in tqdm(scene.getTrainCameras(), desc="render + mask prototypes"):
        torch.cuda.empty_cache()
        view.feature_height, view.feature_width = view.image_height, view.image_width

        rendered = render_contrastive_feature(
            view,
            feature_gaussians,
            pipe,
            background,
            norm_point_features=True,
            multi_lvl_norm=True,
            multi_lvl_dim=multi_lvl_dim,
            smooth_type=None,
        )["render"].permute(1, 2, 0)  

        sam_masks = view.original_masks.cuda().float()  
        sam_masks[sam_masks == -1] = -1000 

        onehot_per_lvl: list[torch.Tensor] = []
        first_id_per_lvl: list[int] = []
        for lvl in range(1, num_lvl + 1) if num_lvl > 1 else [0]:
            tmp = sam_masks[lvl].clone()
            first_id = int(sam_masks[lvl - 1].int().max()) + 1 if lvl > 0 else 0
            first_id_per_lvl.append(first_id)
            tmp -= first_id - 1
            tmp[tmp < 0] = 0 
            onehot = F.one_hot(tmp.long(), num_classes=int(tmp.max().item()) + 1)[:, :, 1:].float()
            onehot_per_lvl.append(onehot)

        masks_per_lvl: list[torch.Tensor] = []
        for lvl in range(num_lvl):
            keep = onehot_per_lvl[lvl].sum(dim=(0, 1)) > MASK_SIZE_THRESH
            masks_per_lvl.append(onehot_per_lvl[lvl][:, :, keep])
            for k in torch.nonzero(keep).flatten().tolist():
                idx_map[lvl].append({"view_name": view.image_name, "mask_id": first_id_per_lvl[lvl] + k})

        segments = []
        start = 0
        for dim in multi_lvl_dim:
            segments.append(F.normalize(rendered[:, :, start : start + dim], dim=-1))
            start += dim
        normed = torch.cat(segments, dim=-1)

        features_per_lvl: list[torch.Tensor] = []
        prefix = 0
        for i in range(num_lvl):
            features_per_lvl.insert(0, F.normalize(normed[:, :, : multi_lvl_dim[i] + prefix], dim=-1))
            prefix += multi_lvl_dim[i]

        for lvl in range(num_lvl):
            feats = features_per_lvl[lvl]
            if feats.shape[:2] != masks_per_lvl[lvl].shape[:2]:
                feats = (
                    F.interpolate(
                        feats.permute(2, 0, 1).unsqueeze(0),
                        masks_per_lvl[lvl].shape[:2],
                        mode="bilinear",
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
            prototypes[lvl].append(
                F.normalize(torch.einsum("hwc,hwf->fc", feats, masks_per_lvl[lvl]), dim=-1, p=2)
            )

    return [torch.cat(p, 0) for p in prototypes], idx_map


def cluster_prototypes(prototypes: list[torch.Tensor]):
    
    all_labels, all_centers = [], []
    for lvl, protos in enumerate(prototypes):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            cluster_selection_epsilon=HDBSCAN_EPS_PER_LEVEL * (lvl + 1),
        )
        labels = clusterer.fit_predict(protos.detach().cpu().numpy())
        ids = [int(i) for i in np.unique(labels) if i != -1]
        if not ids:
            raise SystemExit(f"[cluster] level {lvl}: HDBSCAN found no objects among {len(labels)} masks")
        if ids != list(range(len(ids))):  
            raise SystemExit(f"[cluster] level {lvl}: non-contiguous cluster ids {ids}")

        centers = torch.zeros(len(ids), protos.shape[-1], device=protos.device)
        for i in ids:
            member = torch.from_numpy(labels == i).to(protos.device)
            centers[i] = F.normalize(protos[member].mean(dim=0), dim=-1)

        n_noise = int((labels == -1).sum())
        print(f"[cluster] level {lvl}: {len(labels)} masks -> {len(ids)} objects, {n_noise} noise")
        all_labels.append(labels)
        all_centers.append(centers)
    return all_labels, all_centers


@torch.no_grad()
def point_object_scores(feature_gaussians, centers: list[torch.Tensor], num_lvl: int, multi_lvl_dim: list[int]):
    point_features = feature_gaussians.get_point_features
    segments = []
    start = 0
    for dim in multi_lvl_dim:
        segments.append(F.normalize(point_features[:, start : start + dim], dim=-1))
        start += dim
    normed = torch.cat(segments, dim=-1)

    features_per_lvl: list[torch.Tensor] = []
    prefix = 0
    for i in range(num_lvl):
        features_per_lvl.insert(0, F.normalize(normed[:, : multi_lvl_dim[i] + prefix], dim=-1))
        prefix += multi_lvl_dim[i]

    return [
        torch.einsum("nc,bc->bn", centers[lvl].cuda(), features_per_lvl[lvl].cuda()).cpu()
        for lvl in range(num_lvl)
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Cluster SAM masks into 3D objects (notebook logic)")
    p.add_argument("--dataset", default="causal-lerf")
    p.add_argument("--scene_name", default="ramen")
    p.add_argument("--base_dir", default=str(ROOT), help="casual-main root (contains data/)")
    p.add_argument("--repo_root", default=str(ROOT.parent), help="repo root (contains saved_masks/)")
    p.add_argument("--feature_iter", type=int, default=30000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    scene_path = base_dir / "data" / args.dataset / args.scene_name
    model_path = scene_path / "gs"
    iter_dir = model_path / "point_cloud" / f"iteration_{args.feature_iter}"
    save_dir = Path(args.repo_root).resolve() / "saved_masks" / args.dataset / args.scene_name

    feature_ply = iter_dir / "contrastive_feature_point_cloud.ply"
    if not feature_ply.is_file():
        raise SystemExit(f"[cluster] missing {feature_ply} (run train_affinity_features.py first)")
    save_dir.mkdir(parents=True, exist_ok=True)

    num_lvl = 1 if any(x in args.dataset for x in ("scannet", "3dovs")) else 3
    multi_lvl_dim = [16, 8, 8] if num_lvl == 3 else [32]

    ns, model, pipeline = load_cfg_args(model_path, scene_path)
    dataset = model.extract(ns)
    dataset.need_masks = True

    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
    scene = Scene(
        dataset,
        None,
        feature_gaussians,
        load_iteration=args.feature_iter,
        feature_load_iteration=args.feature_iter,
        shuffle=False,
        mode="eval",
        target="contrastive_feature",
    )
    pipe = pipeline.extract(ns)
    print(f"[cluster] views={len(scene.getTrainCameras())} levels={num_lvl} gaussians={feature_gaussians.get_xyz.shape[0]}")

    prototypes, idx_map = extract_mask_prototypes(scene, feature_gaussians, pipe, num_lvl, multi_lvl_dim)
    for lvl in range(num_lvl):
        assert prototypes[lvl].shape[0] == len(idx_map[lvl]), (lvl, prototypes[lvl].shape, len(idx_map[lvl]))

    labels, centers = cluster_prototypes(prototypes)
    seg_scores = point_object_scores(feature_gaussians, centers, num_lvl, multi_lvl_dim)

    torch.save(seg_scores, iter_dir / "multi_lvl_seg_scores.pth")
    torch.save(labels, save_dir / "multi_lvl_cluster_labels.pth")
    with open(save_dir / "multi_lvl_global_idx_map.json", "w", encoding="utf-8") as f:
        json.dump(idx_map, f)

    print(f"[cluster] wrote {iter_dir / 'multi_lvl_seg_scores.pth'}")
    print(f"[cluster] wrote {save_dir / 'multi_lvl_cluster_labels.pth'}")
    print(f"[cluster] wrote {save_dir / 'multi_lvl_global_idx_map.json'}")


if __name__ == "__main__":
    main()
