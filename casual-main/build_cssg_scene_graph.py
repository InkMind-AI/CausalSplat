
import os
import re
import json
import math
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

def parse_vlm_fields(desc: str) -> Dict[str, any]:
    if not desc:
        return {
            "category": "unknown",
            "appearance": "",
            "affordance": ["unknown_function"],
            "alternative_names": [],
        }

    def extract(pattern, text, default=""):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            val = re.sub(r"[。.,;:!]+$", "", val)
            return val
        return default

    category = extract(r"(?:1\)\s*)?Category:\s*([^\n\r]+)", desc, "unknown").lower()
    appearance = extract(r"(?:2\)\s*)?Appearance:\s*([^\n\r]+)", desc, "")
    affordance_raw = extract(r"(?:3\)\s*)?Affordance:\s*([^\n\r]+)", desc, "unknown_function")
    alt_raw = extract(r"(?:4\)\s*)?Alternative names:\s*([^\n\r]+)", desc, "")

    
    affordance_list = [a.strip() for a in re.split(r"[,;]", affordance_raw) if a.strip()]
    if not affordance_list:
        affordance_list = ["unknown_function"]

    
    alt_list = [a.strip() for a in re.split(r"[,;]", alt_raw) if a.strip()]

    return {
        "category": category if category else "unknown",
        "appearance": appearance,
        "affordance": affordance_list,
        "alternative_names": alt_list,
    }


def axis_overlap_ratio_1d(a_min, a_max, b_min, b_max) -> float:
    inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    a_len = max(1e-8, a_max - a_min)
    return inter / a_len


def xy_overlap_ratio(box_a: Dict, box_b: Dict) -> float:
    
    ax1, ay1 = box_a["bbox_min"][0], box_a["bbox_min"][1]
    ax2, ay2 = box_a["bbox_max"][0], box_a["bbox_max"][1]
    bx1, by1 = box_b["bbox_min"][0], box_b["bbox_min"][1]
    bx2, by2 = box_b["bbox_max"][0], box_b["bbox_max"][1]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(1e-8, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def load_scene_and_xyz(base_dir: Path, dataset: str, scene_name: str, feature_iter: int):
    
    model_path = base_dir / "data" / dataset / scene_name / "gs"

    
    cfg_path = model_path / "cfg_args"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_str = f.read()
    cfg_args = eval(cfg_str)  

    parser = ArgumentParser()
    model_group = ModelParams(parser, sentinel=True)
    pipeline_group = PipelineParams(parser)
    parser.add_argument("--target", default="contrastive_feature")
    parser.add_argument("--iteration", default=-1, type=int)

    args_cmd = parser.parse_args([
        "--model_path", str(model_path),
        "--target", "contrastive_feature",
    ])

    merged = vars(cfg_args).copy()
    for k, v in vars(args_cmd).items():
        if v is not None:
            merged[k] = v
    merged["source_path"] = str(base_dir / "data" / dataset / scene_name)
    merged["model_path"] = str(model_path)

    args = Namespace(**merged)

    dataset_args = model_group.extract(args)
    dataset_args.need_masks = True

    FEATURE_DIM = 32
    gaussians = GaussianModel(dataset_args.sh_degree)
    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
    scene = Scene(
        dataset_args,
        gaussians,
        feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=feature_iter,
        shuffle=False,
        mode="eval",
        target="contrastive_feature",
    )

    xyz = feature_gaussians.get_xyz.detach().cpu()  
    return xyz


def build_nodes(
    xyz: torch.Tensor,
    seg_score: torch.Tensor,  
    vlm_results: Dict[str, str],
    color_masks_dir: Path = None,   
    min_color_mask_imgs: int = 8,
) -> Dict[str, Dict]:
    
    point_to_object = seg_score.argmax(dim=1)  
    n_objects = seg_score.shape[1]

    
    img_count_cache: Dict[str, int] = {}
    if color_masks_dir is not None and color_masks_dir.exists():
        supported = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        for obj_dir in color_masks_dir.iterdir():
            if obj_dir.is_dir():
                cnt = sum(1 for f in obj_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in supported)
                img_count_cache[obj_dir.name] = cnt

    nodes = {}
    skipped = 0
    for obj_id in range(n_objects):
        obj_key = f"object_{obj_id:04d}"

        
        if color_masks_dir is not None:
            n_imgs = img_count_cache.get(obj_key, 0)
            if n_imgs < min_color_mask_imgs:
                skipped += 1
                continue   

        mask = (point_to_object == obj_id)
        n_pts = int(mask.sum().item())
        node_name = f"node_{obj_id}"

        if n_pts == 0:
            center = None
            bbox_min = None
            bbox_max = None
            bbox_size = None
        else:
            pts = xyz[mask]  
            center_t = pts.mean(dim=0)
            bbox_min_t = pts.min(dim=0).values
            bbox_max_t = pts.max(dim=0).values
            size_t = bbox_max_t - bbox_min_t

            center = center_t.tolist()
            bbox_min = bbox_min_t.tolist()
            bbox_max = bbox_max_t.tolist()
            bbox_size = size_t.tolist()

        desc = vlm_results.get(obj_key, "") or ""
        vlm_fields = parse_vlm_fields(desc)

        nodes[node_name] = {
            "object_key": obj_key,
            "category": vlm_fields["category"],
            "appearance": vlm_fields["appearance"],
            "affordance": vlm_fields["affordance"],
            "alternative_names": vlm_fields["alternative_names"],
            "description": desc,
            "num_points": n_pts,
            "center": center,            
            "bbox_min": bbox_min,        
            "bbox_max": bbox_max,        
            "bbox_size": bbox_size,      
        }

    print(f"  : {len(nodes)} ， {skipped} （ < {min_color_mask_imgs} ）")
    return nodes
def build_edges(
    nodes: Dict[str, Dict],
    near_thresh: float = 1.5,
    direction_thresh: float = 0.3,   
) -> List[Dict]:
    
    edges = []
    node_ids = list(nodes.keys())

    
    
    centers = [n["center"] for n in nodes.values() if n.get("center") is not None]
    
    if centers:
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        zs = [c[2] for c in centers]
        scene_dx = max(xs) - min(xs)
        scene_dy = max(ys) - min(ys)
        scene_dz = max(zs) - min(zs)
    else:
        scene_dx = scene_dy = scene_dz = 1.0

    
    
    scene_xy = max(1e-6, math.sqrt(scene_dx * scene_dx + scene_dy * scene_dy))
    scene_yz = max(1e-6, math.sqrt(scene_dy * scene_dy + scene_dz * scene_dz))
    scene_xz = max(1e-6, math.sqrt(scene_dx * scene_dx + scene_dz * scene_dz))

    def safe_size(node: Dict) -> List[float]:
        s = node.get("bbox_size")
        if s is None:
            return [0.0, 0.0, 0.0]
        return [float(abs(s[0])), float(abs(s[1])), float(abs(s[2]))]

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            ni, nj = nodes[node_ids[i]], nodes[node_ids[j]]
            if ni["center"] is None or nj["center"] is None:
                continue

            cx_i, cy_i, cz_i = ni["center"]
            cx_j, cy_j, cz_j = nj["center"]
            d = euclidean(ni["center"], nj["center"])
            if d < 1e-8:
                continue

            dx = cx_i - cx_j   
            dy = cy_i - cy_j   
            dz = cz_i - cz_j   

            perp_z = math.sqrt(dx * dx + dy * dy)   
            perp_x = math.sqrt(dy * dy + dz * dz)   
            perp_y = math.sqrt(dx * dx + dz * dz)   
            sx_i, sy_i, sz_i = safe_size(ni)
            sx_j, sy_j, sz_j = safe_size(nj)
            avg_sx = 0.5 * (sx_i + sx_j)
            avg_sy = 0.5 * (sy_i + sy_j)
            avg_sz = 0.5 * (sz_i + sz_j)
            perp_limit_z = max(0.55 * math.sqrt(avg_sx * avg_sx + avg_sy * avg_sy), 0.08 * scene_xy) 
            perp_limit_x = max(0.55 * math.sqrt(avg_sy * avg_sy + avg_sz * avg_sz), 0.08 * scene_yz)

            perp_limit_y = max(0.55 * math.sqrt(avg_sx * avg_sx + avg_sz * avg_sz), 0.08 * scene_xz)
            perp_limit_z = min(perp_limit_z, 0.35 * scene_xy)
            perp_limit_x = min(perp_limit_x, 0.35 * scene_yz)
            perp_limit_y = min(perp_limit_y, 0.35 * scene_xz)

            if d < near_thresh:
                conf = round(float(max(0.0, 1.0 - d / near_thresh)), 4)
                edges.append({
                    "source": node_ids[i], "target": node_ids[j],
                    "relation": "near", "distance": round(float(d), 4),
                    "confidence": conf
                })
                edges.append({
                    "source": node_ids[j], "target": node_ids[i],
                    "relation": "near", "distance": round(float(d), 4),
                    "confidence": conf
                })

            
            if d > near_thresh:
                continue
         
            conf_dir = round(float(max(0.0, 1.0 - d / max(1e-8, near_thresh))), 4)

            def add_dir(rel_i, rel_j, axis_ratio, perp_dist, perp_limit):
                edges.append({
                    "source": node_ids[i], "target": node_ids[j],
                    "relation": rel_i, "distance": round(float(d), 4),
                    "axis_ratio": round(float(axis_ratio), 4),
                    "perp_dist": round(float(perp_dist), 4),
                    "perp_limit": round(float(perp_limit), 4),
                    "confidence": conf_dir
                })
                edges.append({
                    "source": node_ids[j], "target": node_ids[i],
                    "relation": rel_j, "distance": round(float(d), 4),
                    "axis_ratio": round(float(axis_ratio), 4),
                    "perp_dist": round(float(perp_dist), 4),
                    "perp_limit": round(float(perp_limit), 4),
                    "confidence": conf_dir
                })

            ratio_x = abs(dx) / d
            if ratio_x > direction_thresh and perp_x < perp_limit_x:
                if dx > 0:
                    add_dir("right_of", "left_of", ratio_x, perp_x, perp_limit_x)
                else:
                    add_dir("left_of", "right_of", ratio_x, perp_x, perp_limit_x)

            ratio_z = abs(dz) / d
            if ratio_z > direction_thresh and perp_z < perp_limit_z:
                if dz > 0:
                    add_dir("above", "below", ratio_z, perp_z, perp_limit_z)
                else:
                    add_dir("below", "above", ratio_z, perp_z, perp_limit_z)

            ratio_y = abs(dy) / d
            if ratio_y > direction_thresh and perp_y < perp_limit_y:
                if dy > 0:
                    add_dir("behind", "in_front_of", ratio_y, perp_y, perp_limit_y)
                else:
                    add_dir("in_front_of", "behind", ratio_y, perp_y, perp_limit_y)

    return edges

def serialize_for_llm(scene_graph: Dict,
                      include_metrics: bool = False) -> str:
    lines = []
    lines.append("Nodes:")
    for nid, node in scene_graph["nodes"].items():
        cat  = node.get("category", "unknown")
        aff  = ", ".join(node.get("affordance", []))
        app  = node.get("appearance", "")
        alts = ", ".join(node.get("alternative_names", []))
        cen  = node.get("center", None)
        cen_str = f"({cen[0]:.3f}, {cen[1]:.3f}, {cen[2]:.3f})" if cen else "None"
        
        short_id = re.sub(r"node_0*", "", nid)  
        parts = [f"{short_id}: {cat}"]
        if app:
            parts.append(f"Appearance: {app}")
        if aff:
            parts.append(f"Affordance: {aff}")
        if alts:
            parts.append(f"AKA: {alts}")
        lines.append(". ".join(parts) + ".")

    lines.append("\nTopology:")
    
    KEEP_RELS = {"left_of", "above", "in_front_of"}
    for e in scene_graph["edges"]:
        rel = e["relation"]
        if rel not in KEEP_RELS:
            continue
        src = re.sub(r"node_0*", "", e["source"])
        tgt = re.sub(r"node_0*", "", e["target"])
        if include_metrics:
            d    = e.get("distance", "NA")
            conf = e.get("confidence", "NA")
            lines.append(f"{src} {rel} {tgt} ({d}m,{conf})")
        else:
            d = e.get("distance", "NA")
            d_str = f"{float(d):.2f}" if d != "NA" else "NA"
            lines.append(f"{src} {rel} {tgt} {d_str}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser("Build CSSG scene graph from 3DGS")
    _root = Path(__file__).resolve().parent
    parser.add_argument("--base_dir", type=str, default=str(_root), help="casual-main (data/)")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(_root.parent),
        help="repo root (saved_masks/, vlm_results.json)",
    )
    parser.add_argument("--dataset", type=str, default="causal-lerf")
    parser.add_argument("--scene_name", type=str, default="ramen")
    parser.add_argument("--feature_iter", type=int, default=30000)
    parser.add_argument("--level", type=int, default=2, help="0/1/2")
    parser.add_argument("--near_thresh", type=float, default=0.5)
    parser.add_argument("--direction_thresh", type=float, default=0.3, help="：||/ > ")
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    repo_root = Path(args.repo_root).resolve()

    if args.level == 2:
        if any(x in args.dataset for x in ('scannet',)):
            args.level = 0
        else:
            args.level = 2  
    print(f"[level] dataset={args.dataset} → level={args.level}")
    save_dir = repo_root / "saved_masks" / args.dataset / args.scene_name
    save_dir.mkdir(parents=True, exist_ok=True)

    seg_path = base_dir / "data" / args.dataset / args.scene_name / "gs" / "point_cloud" / f"iteration_{args.feature_iter}" / "multi_lvl_seg_scores.pth"
    vlm_path = save_dir / "vlm_results.json"

    if not seg_path.exists():
        raise FileNotFoundError(f"missing seg scores: {seg_path}")
    if not vlm_path.exists():
        raise FileNotFoundError(f"missing vlm results: {vlm_path}")

    
    print(f"[1/4] load xyz from scene ...")
    xyz = load_scene_and_xyz(base_dir, args.dataset, args.scene_name, args.feature_iter)

    print(f"[2/4] load multi_lvl_seg_scores ...")
    multi_lvl_seg_scores = torch.load(str(seg_path), map_location="cpu")
    if args.level < 0 or args.level >= len(multi_lvl_seg_scores):
        raise ValueError(f"invalid level={args.level}, got {len(multi_lvl_seg_scores)} levels")
    seg_score = multi_lvl_seg_scores[args.level]  

    print(f"[3/4] load vlm descriptions ...")
    with open(vlm_path, "r", encoding="utf-8") as f:
        vlm_results = json.load(f)

    print(f"[4/4] build graph ...")
    color_masks_dir = save_dir / "color_masks"
    nodes = build_nodes(
        xyz, seg_score, vlm_results,
        color_masks_dir=color_masks_dir if color_masks_dir.exists() else None,
        min_color_mask_imgs=8,
    )
    edges = build_edges(
        nodes,
        near_thresh=args.near_thresh,
        direction_thresh=args.direction_thresh,
    )

    scene_graph = {
        "meta": {
            "dataset": args.dataset,
            "scene_name": args.scene_name,
            "feature_iter": args.feature_iter,
            "level": args.level,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "near_thresh": args.near_thresh,
            "direction_thresh": args.direction_thresh,
        },
        "nodes": nodes,
        "edges": edges,
    }

    
    out_json = save_dir / f"scene_graph_lvl{args.level}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scene_graph, f, ensure_ascii=False, indent=2)

    llm_text = serialize_for_llm(scene_graph, include_metrics=False)
    out_txt = save_dir / f"scene_graph_lvl{args.level}_llm.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(llm_text)

    debug_text = serialize_for_llm(scene_graph, include_metrics=True)
    out_debug_txt = save_dir / f"scene_graph_lvl{args.level}_llm_debug.txt"
    with open(out_debug_txt, "w", encoding="utf-8") as f:
        f.write(debug_text)

    print(f"✓ scene graph saved : {out_json}")
    print(f"✓ llm text saved    : {out_txt}")
    print(f"✓ debug text saved  : {out_debug_txt}")


if __name__ == "__main__":
    main()