import os
import re
import sys
import json
from pathlib import Path as _Path

_CAUSALSPLAT_ROOT = _Path(__file__).resolve().parent.parent
if str(_CAUSALSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALSPLAT_ROOT))
import base64
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
def _sanitize_filename(s, max_len=80):
    s = str(s).replace("/", "_").replace("\\", "_")
    s = re.sub(r'[<>:"|?*]', "_", s)
    s = s.strip() or "unnamed"
    return s[:max_len] if len(s) > max_len else s
def save_semantic_slices(model_points, pred_mask, label_str, save_root, subdir=""):
    try:
        import open3d as o3d
    except ImportError:
        print("[ ply ] open3d ")
        return
    model_points = np.asarray(model_points)
    if hasattr(pred_mask, "cpu"):
        pred_mask = pred_mask.cpu().numpy()
    pred_mask = np.asarray(pred_mask).reshape(-1).astype(bool)
    vis_save_root = os.path.join(save_root, "semantic_slices")
    if subdir:
        vis_save_root = os.path.join(vis_save_root, subdir)
    os.makedirs(vis_save_root, exist_ok=True)
    vis_colors = np.ones((len(model_points), 3)) * 0.8
    vis_colors[pred_mask] = [1.0, 0.0, 0.0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(model_points)
    pcd.colors = o3d.utility.Vector3dVector(vis_colors)
    name_part = _sanitize_filename(label_str)
    out_name = f"slice_{name_part}.ply"
    out_path = os.path.abspath(os.path.join(vis_save_root, out_name))
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  [ply] {out_path}")
BASE_DIR = Path(__file__).resolve().parent
try:
    from eval_scannet_semantic import run_scannet_semantic_eval as _run_semantic_eval
    _SEMANTIC_EVAL_AVAILABLE = True
except ImportError:
    _SEMANTIC_EVAL_AVAILABLE = False
FEATURE_GAUSSIAN_ITER = 30000
SUPPORTED_EXTS   = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MIN_IMAGE_COUNT  = 8   
MAX_IMGS_IN_MSG  = 2   
def get_object_images(obj_key: str, color_masks_dir: Path,
                      max_imgs: int = 2) -> List[Path]:
    obj_dir = color_masks_dir / obj_key
    if not obj_dir.exists():
        return []
    imgs = sorted(p for p in obj_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if len(imgs) <= max_imgs:
        return imgs
    step = len(imgs) // max_imgs
    return imgs[::step][:max_imgs]
def crop_foreground(img_path: Path):
    from object_image_utils import encode_object_image
    try:
        encoded, _ = encode_object_image(img_path, extension='.jpg')
        return encoded
    except (OSError, ValueError) as exc:
        print(f"[object crop] {img_path}: {exc}")
        return None
def build_node_image_blocks(scene_graph: Dict[str, Any],
                             color_masks_dir: Path,
                             max_imgs_in_msg: int = MAX_IMGS_IN_MSG) -> List[Dict]:
    blocks = []
    for nid, node in scene_graph.get("nodes", {}).items():
        obj_key = node.get("object_key", "")
        cat= node.get("category", "unknown")
        all_imgs = get_object_images(obj_key, color_masks_dir, max_imgs=9999)
        if len(all_imgs) < MIN_IMAGE_COUNT:
            continue
        sampled = get_object_images(obj_key, color_masks_dir, max_imgs=max_imgs_in_msg)
        if not sampled:
            continue
        blocks.append({"type": "text",
                       "text": f"\n[{nid}] {cat} ({obj_key}):"})
        for img_path in sampled:
            crop_bytes = crop_foreground(img_path)
            if crop_bytes is not None:
                b64 = base64.b64encode(crop_bytes).decode("utf-8")
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                  "detail": "low"}
                })
    return blocks
def build_llm_text_from_json(scene_graph: Dict[str, Any]) -> str:
    lines = ["Nodes:"]
    for nid, node in scene_graph.get("nodes", {}).items():
        cat = node.get("category", "unknown")
        aff = ", ".join(node.get("affordance", []))
        center = node.get("center", None)
        if center is None:
            lines.append(f"[{nid}]: {cat}. Affordance: {aff}. Center: None.")
        else:
            lines.append(
                f"[{nid}]: {cat}. Affordance: {aff}. "
                f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})."
            )
    lines.append("\nTopology:")
    for e in scene_graph.get("edges", []):
        rel = e.get("relation", "unknown")
        s = e.get("source", "unknown")
        t = e.get("target", "unknown")
        if rel == "near":
            lines.append(f"- [{s}] is near [{t}] (distance: {e.get('distance','NA')}m, conf: {e.get('confidence','NA')})")
        elif rel == "supported_by":
            lines.append(
                f"- [{s}] is supported_by [{t}] "
                f"(dz: {e.get('height_gap','NA')}m, overlap_xy: {e.get('xy_overlap_ratio','NA')}, conf: {e.get('confidence','NA')})"
            )
        else:
            lines.append(f"- [{s}] {rel} [{t}]")
    return "\n".join(lines)
def parse_response_to_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"```json\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    m2 = re.search(r"node_(\d+)", raw)
    if m2:
        target = m2.group(1)  
    else:
        m3 = re.search(r'"target_node"\s*:\s*"(\d+)"', raw)
        target = m3.group(1) if m3 else None
    return {
        "anchor_node": None,
        "target_node": target,
        "reasoning": [raw],
        "confidence": None
    }
def resolve_query_json_dir(data_dir: Path) -> Path:
    
    json_root = data_dir / "json"
    test_json = json_root / "test_json"
    if test_json.is_dir() and any(test_json.glob("*.json")):
        return test_json
    if json_root.is_dir() and any(json_root.glob("*.json")):
        return json_root
    raise FileNotFoundError(
        f"No query JSON under {json_root} (expected json/test_json/*.json for Causal-LeRF)"
    )


def extract_queries_ref_lerf(json_dir: Path) -> Dict[str, List[str]]:
    sentence_to_segs: Dict[str, List[str]] = {}
    if not json_dir.exists():
        return sentence_to_segs
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[]  {json_file} : {e}")
            continue
        for obj in data.get("object", []):
            seg = obj.get("segmentation", "").strip()
            sentences = obj.get("sentence", [])
            if not seg or not sentences:
                continue
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if s not in sentence_to_segs:
                    sentence_to_segs[s] = []
                if seg not in sentence_to_segs[s]:
                    sentence_to_segs[s].append(seg)
    return sentence_to_segs
def extract_queries_lerf_ovs(gt_dir: Path) -> List[str]:
    vocabs = set()
    for frame_dir in sorted(gt_dir.iterdir()):
        if not frame_dir.is_dir():
            continue
        for img_file in frame_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                vocabs.add(img_file.stem)
    return sorted(vocabs)
def load_scene_and_gaussians(dataset: str, scene_name: str):
    sys.path.insert(0, str(BASE_DIR))
    from arguments import ModelParams, PipelineParams
    from scene import Scene, GaussianModel, FeatureGaussianModel
    from argparse import ArgumentParser, Namespace
    model_path = str(BASE_DIR / "data" / dataset / scene_name / "gs")
    cfg_path = os.path.join(model_path, "cfg_args")
    with open(cfg_path) as f:
        cfg_str = f.read()
    cfg_args = eval(cfg_str)
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--target", default="contrastive_feature")
    parser.add_argument("--iteration", default=-1, type=int)
    args_cmd = parser.parse_args(["--model_path", model_path, "--target", "contrastive_feature"])
    merged = vars(cfg_args).copy()
    for k, v in vars(args_cmd).items():
        if v is not None:
            merged[k] = v
    merged["source_path"] = str(BASE_DIR / "data" / dataset / scene_name)
    merged["model_path"] = model_path
    args = Namespace(**merged)
    dataset_args = mp.extract(args)
    dataset_args.need_masks = True
    scene_gaussians = GaussianModel(dataset_args.sh_degree)
    feature_gaussians = FeatureGaussianModel(32)
    scene = Scene(
        dataset_args, scene_gaussians, feature_gaussians,
        load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITER,
        shuffle=False, mode="eval", target="contrastive_feature",
    )
    pipeline = pp.extract(args)
    return scene, scene_gaussians, pipeline
def get_object_point_mask(seg_score: torch.Tensor, object_id: int) -> torch.Tensor:
    point_to_object = seg_score.argmax(dim=1)
    return (point_to_object == object_id)
def render_object_mask(view, scene_gaussians, pipeline, point_mask: torch.Tensor) -> np.ndarray:
    from gaussian_renderer import render
    N = scene_gaussians.get_xyz.shape[0]
    colors = torch.zeros(N, 3, device="cuda")
    colors[point_mask] = 1.0
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        res = render(view, scene_gaussians, pipeline, bg, override_color=colors)
    rendered = res["render"]
    return (rendered[0] > 0.5).cpu().numpy().astype(np.uint8)
def render_object_vis(view, scene_gaussians, pipeline, point_mask: torch.Tensor) -> np.ndarray:
    from gaussian_renderer import render
    bg_black = torch.zeros(3, dtype=torch.float32, device="cuda")
    bg_white = torch.ones(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        rgb_res = render(view, scene_gaussians, pipeline, bg_black)
        rgb = rgb_res["render"].permute(1, 2, 0).cpu().numpy()  
        N = scene_gaussians.get_xyz.shape[0]
        colors = torch.zeros(N, 3, device="cuda")
        colors[point_mask] = 1.0
        mask_res = render(view, scene_gaussians, pipeline, bg_black, override_color=colors)
        mask = (mask_res["render"][0] > 0.5).cpu().numpy()  
    vis = np.ones_like(rgb)  
    vis[mask] = rgb[mask]
    vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis_bgr
def load_gt_mask(gt_img_path: Path, target_h: int, target_w: int) -> np.ndarray:
    img = cv2.imread(str(gt_img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f" gt : {gt_img_path}")
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (img.max(axis=2) > 10).astype(np.uint8)
def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else float("nan")
def call_llm(client, model: str, graph_text: str, query: str,
             max_tokens: int, temperature: float,
             node_image_blocks: List[Dict] = None) -> tuple:
    system_prompt = (
        "You are an advanced cognitive module for an embodied robot agent. "
        "Your task is to identify the target object node from a 3D scene graph given a natural language instruction.\n"
        "Scene graph format:\n"
        "- Nodes: each line is 'ID: category. Appearance: ... Affordance: ... AKA: ...'\n"
        "- Topology edges: each line is 'A rel B', meaning A has relation rel to B. "
        "Relations are one-directional and use only: left_of, above, in_front_of. "
        "Their inverses can be inferred: left_of <-> right_of, above <-> below, in_front_of <-> behind. "
        "Example: '5 left_of 9' means node 5 is to the LEFT of node 9, so node 9 is to the RIGHT of node 5. "
        "'5 above 9' means node 5 is ABOVE node 9, so node 9 is BELOW node 5. "
        "'5 in_front_of 9' means node 5 is IN FRONT OF node 9, so node 9 is BEHIND node 5.\n"
        "Follow this reasoning pipeline:\n"
        "1. Intent Analysis: Parse the instruction to identify what the user wants and what cues are given "
        "(semantic description, spatial/relational cues, affordance cues, etc.).\n"
        "2. Spatial Check: Determine whether the instruction contains explicit spatial or topological "
        "relational cues (e.g., 'next to', 'on top of', 'near', 'beside', 'behind'). "
        "If NO spatial cues are present, skip steps 3-4 and directly match the target node by "
        "semantic category, appearance description, or affordance.\n"
        "3. Anchor Localization (only if spatial cues exist): Find the anchor node that the instruction "
        "references as a spatial reference point (e.g., 'next to the rubber duck' -> anchor = rubber duck node).\n"
        "4. Topology Search (only if spatial cues exist): Using the Topology edges (and their inferred inverses), "
        "find the node that satisfies both the spatial relation to the anchor and the semantic description.\n"
        "5. Final Selection: Output the single best matching target node ID (number only, e.g. 42).\n"
        "Edge format: 'A rel B D', where D is the distance in meters between A and B.\n"
        "Output ONLY valid JSON. Do not output any explanation outside the JSON."
    )
    text_block = f"""[Scene Graph]
{graph_text}
[User Instruction]
\"{query}\"
[Output Format — strict JSON only]
{{
  \"has_spatial_cue\": true,
  \"anchor_node\": \"node_x or null\",
  \"target_node\": \"node_y or null\",
  \"reasoning\": [\"step 1\", \"step 2\", \"step 3\"],
  \"confidence\": 0.0
}}""".strip()
    if node_image_blocks:
        user_content = [
            {"type": "text",
             "text": f"[Scene Graph]\n{graph_text}\n\n[Node Visual References (images per node)]:"}
        ] + node_image_blocks + [
            {"type": "text",
             "text": f"\n[User Instruction]\n\"{query}\"\n\n[Output Format — strict JSON only]\n{{"
                     f'\n  \"has_spatial_cue\": true,'
                     f'\n  \"anchor_node\": \"node_x or null\",'
                     f'\n  \"target_node\": \"node_y or null\",'
                     f'\n  \"reasoning\": [\"step 1\", \"step 2\", \"step 3\"],'
                     f'\n  \"confidence\": 0.0'
                     f'\n}}'}
        ]
    else:
        user_content = text_block
        print("，")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(os.environ.get("VLM_REQUEST_TIMEOUT_S", "120")),
        )
        raw = resp.choices[0].message.content
        return raw, parse_response_to_json(raw)
    except Exception as e:
        err_str = str(e)
        if "400" in err_str and "input length" in err_str.lower():
            print(f"  [LLM  - ，] {e}")
            if node_image_blocks:  
                try:
                    resp2 = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": text_block},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=float(os.environ.get("VLM_REQUEST_TIMEOUT_S", "120")),
                    )
                    raw2 = resp2.choices[0].message.content
                    print(f"  [（）]")
                    return raw2, parse_response_to_json(raw2)
                except Exception as e2:
                    print(f"  [] {e2}")
                    return "", {"target_node": None, "error": str(e2)}
        print(f"  [LLM ] {e}")
        return "", {"target_node": None, "error": err_str}
def node_id_to_object_id(node_id: str) -> Optional[int]:
    m = re.search(r"(\d+)", node_id or "")
    return int(m.group(1)) if m else None
def normalize_target_node(target_node_raw: Any) -> Optional[str]:
    if target_node_raw is None:
        return None
    tn = str(target_node_raw).strip()
    if not tn:
        return None
    return tn if tn.startswith("node_") else f"node_{tn}"
def main():
    parser = argparse.ArgumentParser("CSSG + LLM reasoning with mIoU evaluation")
    parser.add_argument("--base_dir", type=str, default=str(BASE_DIR), help="casual-main (data/)")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(BASE_DIR.parent),
        help="repo root (saved_masks/, scene graphs)",
    )
    parser.add_argument("--dataset",    type=str, default="causal-lerf")
    parser.add_argument("--scene_name", type=str, default="teatime")
    parser.add_argument("--level",      type=int, default=2)
    from vlm_client import add_vlm_arguments, make_vlm_client

    add_vlm_arguments(parser)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature",type=float, default=0.1)
    parser.add_argument("--skip_eval",  action="store_true", help=" LLM ， IoU")
    args = parser.parse_args()
    if args.level == 2:  
        if any(x in args.dataset for x in ('scannet',)):
            args.level = 0
        else:
            args.level = 2  
    print(f"[level] dataset={args.dataset} → level={args.level}")
    try:
        import openai  
    except ImportError:
        raise RuntimeError(" openai: pip install openai")
    base_dir = Path(args.base_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    dataset    = args.dataset
    scene_name = args.scene_name
    save_dir   = repo_root / "saved_masks" / dataset / scene_name
    data_dir   = base_dir / "data" / dataset / scene_name
    graph_json_path = save_dir / f"scene_graph_lvl{args.level}.json"
    graph_txt_path  = save_dir / f"scene_graph_lvl{args.level}_llm.txt"
    if not graph_json_path.exists():
        raise FileNotFoundError(f": {graph_json_path}")
    scene_graph = json.loads(graph_json_path.read_text(encoding="utf-8"))
    graph_text = graph_txt_path.read_text(encoding="utf-8") if graph_txt_path.exists() else build_llm_text_from_json(scene_graph)
    node_to_obj_id: Dict[str, int] = {}
    for nid, node in scene_graph.get("nodes", {}).items():
        m = re.search(r"(\d+)", node.get("object_key", ""))
        if m:
            node_to_obj_id[nid] = int(m.group(1))
    seg_scores_path = (base_dir / "data" / dataset / scene_name / "gs"
                       / "point_cloud" / f"iteration_{FEATURE_GAUSSIAN_ITER}"
                       / "multi_lvl_seg_scores.pth")
    if not args.skip_eval and seg_scores_path.exists():
        print(f" seg_scores: {seg_scores_path}")
        multi_lvl_seg_scores = torch.load(str(seg_scores_path), map_location="cpu")
        seg_score = multi_lvl_seg_scores[args.level]
        print(f" ...")
        scene, scene_gaussians, pipeline = load_scene_and_gaussians(dataset, scene_name)
        train_cameras = scene.getTrainCameras()
        test_cameras  = scene.getTestCameras()
        all_cameras   = list(train_cameras) + list(test_cameras)
        cam_dict      = {cam.image_name: cam for cam in all_cameras}
        need_render = True
    else:
        seg_score = None
        cam_dict  = {}
        scene_gaussians = None
        pipeline = None
        need_render = False
    client, args.model = make_vlm_client(
        args.api_key,
        args.base_url,
        args.model,
        wait=args.wait_vlm,
        wait_timeout_s=args.wait_vlm_timeout,
    )
    color_masks_dir = save_dir / "color_masks"
    node_image_blocks = None
    if color_masks_dir.exists():
        node_count = len(scene_graph.get("nodes", {}))
        imgs_per_node = MAX_IMGS_IN_MSG if node_count < 62 else 1
        print(f" (={node_count}, {imgs_per_node}) ...")
        node_image_blocks = build_node_image_blocks(scene_graph, color_masks_dir, imgs_per_node)
        n_img_nodes = sum(1 for b in node_image_blocks if b.get("type") == "image_url")
        print(f"  : {n_img_nodes}")
    results      = []
    iou_results: Dict[str, List[float]] = {}
    per_frame_detail = []
    if dataset in ("Ref-lerf", "causal-lerf"):
        json_dir = resolve_query_json_dir(data_dir)
        print(f"[reason] query JSON dir: {json_dir}")
        sentence_to_segs = extract_queries_ref_lerf(json_dir)
        if not sentence_to_segs:
            raise ValueError(f" queries，: {json_dir}")
        print(f" {len(sentence_to_segs)}  queries")
        vocab_to_obj = {}
        vocab_to_obj_path = save_dir / "vocab_to_objects.json"
        if vocab_to_obj_path.exists():
            vocab_to_obj = json.loads(vocab_to_obj_path.read_text(encoding="utf-8"))
        for q_idx, (sentence, seg_paths) in enumerate(sentence_to_segs.items(), 1):
            print(f"\n[{q_idx}/{len(sentence_to_segs)}] {sentence[:70]}")
            raw, parsed = call_llm(client, args.model, graph_text, sentence,
                                   args.max_tokens, args.temperature,
                                   node_image_blocks=node_image_blocks)  
            target_node = normalize_target_node(parsed.get("target_node"))
            result_item = {
                "query_idx":   q_idx,      
                "query_text":  sentence,   
                "result":      parsed,     
                "raw":         raw,        
                }
            if need_render and target_node and target_node in node_to_obj_id:
                obj_id = node_to_obj_id[target_node]
                point_mask = get_object_point_mask(seg_score, obj_id).cuda()
                if point_mask.sum() == 0:
                    print(f"  [{target_node}] obj_{obj_id:04d} ，IoU=0")
                mask_root = data_dir / "mask"
                for seg_path_str in seg_paths:
                    seg_path = mask_root / seg_path_str
                    frame_name = Path(seg_path_str).parent.name
                    cam = cam_dict.get(frame_name)
                    if cam is None:
                        frame_num = int(re.search(r"(\d+)", frame_name).group(1))
                        for k, v in cam_dict.items():
                            m2 = re.search(r"(\d+)", k)
                            if m2 and int(m2.group(1)) == frame_num:
                                cam = v
                                break
                    if cam is None:
                        print(f"  []  {frame_name}，IoU=0")
                        iou_results.setdefault(sentence, []).append(0.0)
                        per_frame_detail.append({"frame": frame_name, "query": sentence,
                                                  "target_node": target_node, "iou": 0.0})
                        continue
                    pred_mask = render_object_mask(cam, scene_gaussians, pipeline, point_mask)
                    try:
                        gt_mask = load_gt_mask(seg_path, pred_mask.shape[0], pred_mask.shape[1])
                    except Exception as e:
                        print(f"  []  GT : {e}，IoU=0")
                        iou_results.setdefault(sentence, []).append(0.0)
                        per_frame_detail.append({"frame": frame_name, "query": sentence,
                                                  "target_node": target_node, "iou": 0.0})
                        continue
                    iou = compute_iou(pred_mask, gt_mask)
                    flag = "" if np.isnan(iou) or iou >= 0.5 else "  ←"
                    print(f"  {sentence[:50]:50s} | frame={frame_name} "
                          f"pred={pred_mask.sum():6d} gt={gt_mask.sum():6d} "
                          f"IoU={iou:.4f}{flag}")
                    iou_results.setdefault(sentence, []).append(iou if not np.isnan(iou) else 0.0)
                    per_frame_detail.append({"frame": frame_name, "query": sentence,
                                              "target_node": target_node, "iou": round(float(iou), 6)})
                    result_item.setdefault("ious", []).append({"frame": frame_name, "iou": round(float(iou), 6)})
                    try:
                        vis_img = render_object_vis(cam, scene_gaussians, pipeline, point_mask)
                        safe_query = re.sub(r'[^\w\s-]', '', sentence[:40]).strip().replace(' ', '_')
                        vis_dir = save_dir / "pred_vis" / f"{q_idx:04d}_{safe_query}"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        vis_path = vis_dir / f"{frame_name}.jpg"
                        cv2.imwrite(str(vis_path), vis_img)
                    except Exception as e_vis:
                        print(f"  [] {e_vis}")
            elif need_render:
                print(f"  [LLM ] target_node={target_node}，IoU=0")
                for seg_path_str in seg_paths:
                    frame_name = Path(seg_path_str).parent.name
                    iou_results.setdefault(sentence, []).append(0.0)
                    per_frame_detail.append({"frame": frame_name, "query": sentence,
                                              "target_node": target_node, "iou": 0.0})
            results.append(result_item)
    elif dataset == "causal-scannet":
        _text_dir = data_dir / f"{scene_name}_text"
        gt_json_candidates = sorted(_text_dir.glob("*.json")) if _text_dir.exists() else []
        if not gt_json_candidates:
            raise FileNotFoundError(f" causal-scannet GT JSON: {_text_dir}")
        gt_json_path = gt_json_candidates[0]
        gt_data = json.loads(gt_json_path.read_text(encoding="utf-8"))
        gt_info = gt_data.get("gt_info", [])
        if not gt_info:
            raise ValueError(f"GT JSON  gt_info: {gt_json_path}")
        labels_file = gt_data.get("file") or f"{scene_name}_vh_clean_2.labels.ply"
        labels_ply_path = data_dir / labels_file
        if not labels_ply_path.exists():
            raise FileNotFoundError(f" labels.ply: {labels_ply_path}")
        print(f"GT JSON : {gt_json_path}")
        print(f"Labels  : {labels_ply_path}")
        print(f" {len(gt_info)}  queries")
        per_gaussian_gt_label = None
        if need_render:
            gs_ply_path = (base_dir / "data" / dataset / scene_name / "gs"
                           / "point_cloud" / f"iteration_{FEATURE_GAUSSIAN_ITER}"
                           / "scene_point_cloud.ply")
            if gs_ply_path.exists():
                try:
                    from plyfile import PlyData as _PlyData
                    _gpv = _PlyData.read(str(gs_ply_path))["vertex"].data
                    gaussian_xyz_np = np.vstack([
                        _gpv["x"].astype(np.float32),
                        _gpv["y"].astype(np.float32),
                        _gpv["z"].astype(np.float32),
                    ]).T
                    _lpv = _PlyData.read(str(labels_ply_path))["vertex"].data
                    labels_label_np = np.array(_lpv["label"], dtype=np.int64)
                    if len(labels_label_np) != len(gaussian_xyz_np):
                        print(f"[] labels.ply ({len(labels_label_np)}) != gaussian ({len(gaussian_xyz_np)})，， 3D IoU")
                    else:
                        per_gaussian_gt_label = labels_label_np  
                        print(f": {len(gaussian_xyz_np)} ")
                except Exception as _e:
                    print(f"[] KNN : {_e}， 3D IoU")
            else:
                print(f"[]  gaussian ply， 3D IoU: {gs_ply_path}")
        for q_idx, obj in enumerate(gt_info, 1):
            label_list = obj.get("label", [])
            if not label_list:
                continue
            referring_expr = str(label_list[0]).strip()
            gt_id_list = obj.get("gt_id", [])
            valid_gt_ids = set(
                int(x) for x in gt_id_list if x is not None and int(x) >= 0
            )
            print(f"\n[{q_idx}/{len(gt_info)}] {referring_expr[:70]}")
            raw, parsed = call_llm(
                client, args.model, graph_text, referring_expr,
                args.max_tokens, args.temperature,
                node_image_blocks=node_image_blocks,
            )
            target_node = normalize_target_node(parsed.get("target_node"))
            result_item = {
                "query_idx":  q_idx,
                "query_text": referring_expr,
                "result":     parsed,
                "raw":        raw,
            }
            if need_render and per_gaussian_gt_label is not None:
                if target_node and target_node in node_to_obj_id:
                    obj_id = node_to_obj_id[target_node]
                    pred_mask_np = (
                        get_object_point_mask(seg_score, obj_id)
                        .cpu().numpy().astype(bool)
                    )  
                else:
                    pred_mask_np = np.zeros(len(per_gaussian_gt_label), dtype=bool)
                gt_mask_np = np.isin(per_gaussian_gt_label, list(valid_gt_ids))
                n_pred = len(pred_mask_np)
                n_gt   = len(gt_mask_np)
                if n_pred != n_gt:
                    print(f"  [] pred ({n_pred}) != gt ({n_gt})，"
                          f"seg_score  scene_point_cloud.ply ， query IoU")
                    iou_results.setdefault(referring_expr, []).append(0.0)
                    per_frame_detail.append({
                        "query":       referring_expr,
                        "target_node": target_node,
                        "gt_cnt":      n_gt,
                        "pred_cnt":    n_pred,
                        "iou_3d":      0.0,
                        "error":       f"point count mismatch: pred={n_pred} gt={n_gt}",
                    })
                    result_item["iou_3d"] = 0.0
                    result_item["error"] = f"point count mismatch: pred={n_pred} gt={n_gt}"
                    results.append(result_item)
                    continue
                print(f"  [] pred ={n_pred}, gt ={n_gt}")
                gt_cnt   = int(gt_mask_np.sum())
                pred_cnt = int(pred_mask_np.sum())
                if gt_cnt == 0 and pred_cnt == 0:
                    iou_3d = 1.0
                else:
                    _inter = np.logical_and(gt_mask_np, pred_mask_np).sum()
                    _union = np.logical_or(gt_mask_np, pred_mask_np).sum()
                    iou_3d = float(_inter) / float(_union) if _union > 0 else 0.0
                flag = "" if iou_3d >= 0.5 else "  ←"
                print(f"  {referring_expr[:50]:50s} | "
                      f"GT:{gt_cnt:<6} Pred:{pred_cnt:<6} IoU:{iou_3d:.4f}{flag}")
                iou_results.setdefault(referring_expr, []).append(iou_3d)
                per_frame_detail.append({
                    "query":       referring_expr,
                    "target_node": target_node,
                    "gt_cnt":      gt_cnt,
                    "pred_cnt":    pred_cnt,
                    "iou_3d":      round(iou_3d, 6),
                })
                result_item["iou_3d"] = round(iou_3d, 6)
                try:
                    save_semantic_slices(
                        gaussian_xyz_np, pred_mask_np,
                        label_str=f"{q_idx:04d}_{referring_expr[:40]}",
                        save_root=str(save_dir),
                        subdir=scene_name,
                    )
                except Exception as _e_vis:
                    print(f"  [] {_e_vis}")
            elif need_render:
                print(f"  [ 3D IoU] target_node={target_node}")
                iou_results.setdefault(referring_expr, []).append(0.0)
            results.append(result_item)
    elif dataset == "scannet":
        from eval_scannet_semantic import (
            CLASS_CONFIGS, NYU40_DICT, read_labels_ply,
            knn_label_transfer, remap_labels, calculate_metrics, save_semantic_ply
        )
        _num_classes_list = [19, 15, 10]
        _all_target_ids = sorted(set(
            tid for nc in _num_classes_list
            for tid in CLASS_CONFIGS.get(nc, [])
        ))
        _vocabs = [NYU40_DICT[tid] for tid in _all_target_ids]  
        print(f"ScanNet : {len(_vocabs)} ")
        _gs_ply_path = (base_dir / "data" / dataset / scene_name / "gs"
                        / "point_cloud" / f"iteration_{FEATURE_GAUSSIAN_ITER}"
                        / "scene_point_cloud.ply")
        _labels_ply_path = data_dir / f"{scene_name}_vh_clean_2.labels.ply"
        _per_gaussian_nyu40 = None
        _gaussian_xyz_np    = None
        if need_render and _gs_ply_path.exists() and _labels_ply_path.exists():
            try:
                from plyfile import PlyData as _PlyData2
                _gpv2 = _PlyData2.read(str(_gs_ply_path))["vertex"].data
                _gaussian_xyz_np = np.vstack([
                    _gpv2["x"].astype(np.float32),
                    _gpv2["y"].astype(np.float32),
                    _gpv2["z"].astype(np.float32),
                ]).T
                print(f": {len(_gaussian_xyz_np)}")
                _mesh_xyz, _mesh_labels = read_labels_ply(_labels_ply_path)
                if len(_mesh_labels) != len(_gaussian_xyz_np):
                    print(f"[] labels.ply ({len(_mesh_labels)}) != gaussian ({len(_gaussian_xyz_np)})，，")
                else:
                    _per_gaussian_nyu40 = _mesh_labels
                    print(f": {len(_gaussian_xyz_np)} ，NYU40 : {np.unique(_per_gaussian_nyu40).tolist()}")
            except Exception as _e_knn:
                print(f"[] KNN : {_e_knn}")
        elif need_render:
            print(f"[]  gaussian ply  labels.ply， 3D IoU")
        _cls_pred_masks: Dict[int, np.ndarray] = {}
        for q_idx, (tid, vocab) in enumerate(zip(_all_target_ids, _vocabs), 1):
            print(f"\n[{q_idx}/{len(_vocabs)}] {vocab} (NYU40 id={tid})")
            raw, parsed = call_llm(
                client, args.model, graph_text, vocab,
                args.max_tokens, args.temperature,
                node_image_blocks=node_image_blocks,
            )
            target_node = normalize_target_node(parsed.get("target_node"))
            result_item = {
                "query_idx":  q_idx,
                "query_text": vocab,
                "nyu40_id":   tid,
                "result":     parsed,
                "raw":        raw,
            }
            if need_render and _per_gaussian_nyu40 is not None:
                if target_node and target_node in node_to_obj_id:
                    obj_id = node_to_obj_id[target_node]
                    pred_mask_np = (
                        get_object_point_mask(seg_score, obj_id)
                        .cpu().numpy().astype(bool)
                    )
                else:
                    print(f"  [LLM ] target_node={target_node}")
                    pred_mask_np = np.zeros(len(_per_gaussian_nyu40), dtype=bool)
                if len(pred_mask_np) != len(_per_gaussian_nyu40):
                    print(f"  [] : pred={len(pred_mask_np)} gt={len(_per_gaussian_nyu40)}，")
                else:
                    _cls_pred_masks[tid] = pred_mask_np
                    gt_mask_np = (_per_gaussian_nyu40 == tid)
                    gt_cnt   = int(gt_mask_np.sum())
                    pred_cnt = int(pred_mask_np.sum())
                    _inter = np.logical_and(gt_mask_np, pred_mask_np).sum()
                    _union = np.logical_or(gt_mask_np, pred_mask_np).sum()
                    iou_cls = float(_inter) / float(_union) if _union > 0 else 0.0
                    flag = "" if iou_cls >= 0.5 else "  ←"
                    print(f"  {vocab:20s} | GT:{gt_cnt:<6} Pred:{pred_cnt:<6} IoU:{iou_cls:.4f}{flag}")
                    iou_results.setdefault(vocab, []).append(iou_cls)
                    result_item["iou_cls"] = round(iou_cls, 6)
                    result_item["gt_cnt"]  = gt_cnt
                    result_item["pred_cnt"] = pred_cnt
            results.append(result_item)
        if need_render and _per_gaussian_nyu40 is not None and _gaussian_xyz_np is not None:
            _vis_dir = save_dir / "semantic_eval" / scene_name
            _vis_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "="*60)
            print("[ScanNet  - LLM ]")
            print("="*60)
            _summary_results = {}
            for num_cls in _num_classes_list:
                target_ids = CLASS_CONFIGS.get(num_cls, [])
                per_cls_ious = []
                pred_nyu40_arr = np.zeros(len(_per_gaussian_nyu40), dtype=np.int64)
                for tid in target_ids:
                    if tid in _cls_pred_masks:
                        pred_nyu40_arr[_cls_pred_masks[tid]] = tid
                gt_remapped   = remap_labels(_per_gaussian_nyu40, target_ids)
                pred_remapped = remap_labels(pred_nyu40_arr,      target_ids)
                gt_tensor   = torch.from_numpy(gt_remapped.astype(np.int64))
                pred_tensor = torch.from_numpy(pred_remapped.astype(np.int64))
                ious, mean_iou, acc, mean_acc = calculate_metrics(
                    gt_tensor, pred_tensor, total_classes=num_cls + 1
                )
                print(f"\n  ── {num_cls} : mIoU={mean_iou:.4f}  oAcc={acc:.4f}  mAcc={mean_acc:.4f}")
                for i, tid in enumerate(target_ids, 1):
                    cls_name = NYU40_DICT.get(tid, str(tid))
                    print(f"    [{tid:2d}] {cls_name:20s}: IoU={ious[i].item():.4f}")
                _ply_path = _vis_dir / f"semantic_{num_cls}cls_llm.ply"
                save_semantic_ply(_gaussian_xyz_np, pred_remapped, gt_remapped, target_ids, _ply_path)
                _summary_results[f"cls{num_cls}"] = {
                    "num_classes": num_cls,
                    "mIoU": round(mean_iou, 6),
                    "oAcc": round(acc, 6),
                    "mAcc": round(mean_acc, 6),
                    "per_class_iou": {
                        NYU40_DICT.get(tid, str(tid)): round(ious[i].item(), 6)
                        for i, tid in enumerate(target_ids, 1)
                    },
                }
            _out_json = _vis_dir / "semantic_eval_llm_summary.json"
            _out_json.write_text(
                json.dumps({"scene_name": scene_name, "results": _summary_results},
                           ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n  [ JSON] {_out_json}")
    else:
        gt_dir = base_dir / "data" / "lerf-ovs" / "label" / scene_name / "gt"
        vocabs = extract_queries_lerf_ovs(gt_dir)
        print(f" {len(vocabs)}  vocabs")
        for q_idx, vocab in enumerate(vocabs, 1):
            print(f"\n[{q_idx}/{len(vocabs)}] {vocab}")
            raw, parsed = call_llm(client, args.model, graph_text, vocab,
                                   args.max_tokens, args.temperature,
                                   node_image_blocks=node_image_blocks)
            target_node = normalize_target_node(parsed.get("target_node"))
            result_item = {"query_idx": q_idx, "query_text": vocab,
                           "result": parsed, "raw": raw}
            if need_render and target_node and target_node in node_to_obj_id:
                obj_id = node_to_obj_id[target_node]
                point_mask = get_object_point_mask(seg_score, obj_id).cuda()
                for frame_dir in sorted(gt_dir.iterdir()):
                    if not frame_dir.is_dir():
                        continue
                    frame_name = frame_dir.name
                    gt_candidates = list(frame_dir.glob(f"{vocab}.*"))
                    if not gt_candidates:
                        continue
                    gt_path = gt_candidates[0]
                    cam = cam_dict.get(frame_name)
                    if cam is None:
                        print(f"  []  {frame_name}，IoU=0")
                        iou_results.setdefault(vocab, []).append(0.0)
                        per_frame_detail.append({"frame": frame_name, "query": vocab,
                                                  "target_node": target_node, "iou": 0.0})
                        continue
                    pred_mask = render_object_mask(cam, scene_gaussians, pipeline, point_mask)
                    try:
                        gt_mask = load_gt_mask(gt_path, pred_mask.shape[0], pred_mask.shape[1])
                    except Exception as e:
                        print(f"  []  GT : {e}")
                        continue
                    iou = compute_iou(pred_mask, gt_mask)
                    flag = "" if np.isnan(iou) or iou >= 0.5 else "  ←"
                    print(f"  {vocab:30s} | frame={frame_name} "
                          f"pred={pred_mask.sum():6d} gt={gt_mask.sum():6d} "
                          f"IoU={iou:.4f}{flag}")
                    iou_results.setdefault(vocab, []).append(iou if not np.isnan(iou) else 0.0)
                    per_frame_detail.append({"frame": frame_name, "query": vocab,
                                              "target_node": target_node, "iou": round(float(iou), 6)})
                    result_item.setdefault("ious", []).append({"frame": frame_name, "iou": round(float(iou), 6)})
                    try:
                        vis_img = render_object_vis(cam, scene_gaussians, pipeline, point_mask)
                        safe_query = re.sub(r'[^\w\s-]', '', vocab[:40]).strip().replace(' ', '_')
                        vis_dir = save_dir / "pred_vis" / f"{q_idx:04d}_{safe_query}"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        vis_path = vis_dir / f"{frame_name}.jpg"
                        cv2.imwrite(str(vis_path), vis_img)
                    except Exception as e_vis:
                        print(f"  [] {e_vis}")
            elif need_render:
                if target_node is None:
                    print(f"  [LLM  null] target_node=None，IoU=0")
                else:
                    print(f"  [LLM ] target_node={target_node}（），IoU=0")
                for frame_dir in sorted(gt_dir.iterdir()):
                    if not frame_dir.is_dir():
                        continue
                    frame_name = frame_dir.name
                    gt_candidates = list(frame_dir.glob(f"{vocab}.*"))
                    if not gt_candidates:
                        continue
                    iou_results.setdefault(vocab, []).append(0.0)
                    per_frame_detail.append({"frame": frame_dir.name, "query": vocab,
                                              "target_node": target_node, "iou": 0.0})
            results.append(result_item)
    if dataset == "scannet":
        print("\n" + "="*60)
        print("[scannet] ， LLM query ")
        print("="*60)
        overall = float("nan")
        per_query_mean_ious = []
        all_ious_flat = []
    else:
        print("\n" + "="*60)
        print("： query  IoU")
        print("="*60)
        per_query_mean_ious = []
        for vocab, ious in sorted(iou_results.items()):
            mean_iou = float(np.mean(ious)) if ious else 0.0
            flag = "  ✓" if mean_iou >= 0.5 else "  ✗"
            print(f"  {vocab[:50]:50s}  mIoU={mean_iou:.4f}  (n={len(ious)}){flag}")
            per_query_mean_ious.append(mean_iou)
        all_ious_flat = [i for ious in iou_results.values() for i in ious]
        overall = float(np.mean(all_ious_flat)) if all_ious_flat else float("nan")
        print(f"\n  Overall mIoU = {overall:.4f}  ( {len(all_ious_flat)} )")
        print("="*60)
    output = {
        "dataset":     dataset,
        "scene_name":  scene_name,
        "level":       args.level,
        "num_queries": len(results),
        "overall_miou": overall,
        "num_evaluated": len(all_ious_flat),
        "per_vocab_mean": {
            v: float(np.mean(ious)) for v, ious in iou_results.items()
        },
        "per_frame": per_frame_detail,
        "results": results,
    }
    out_path = save_dir / f"llm_reasoning_results_lvl{args.level}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n: {out_path}")
if __name__ == "__main__":
    main()
