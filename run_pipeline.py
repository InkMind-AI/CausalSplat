
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR    = Path(__file__).resolve().parent
SAVED_MASKS = BASE_DIR / "saved_masks"



def get_paths(dataset: str, scene_name: str):
    saved       = SAVED_MASKS / dataset / scene_name
    color_masks = saved / "color_masks"
    vlm_results = saved / "vlm_results.json"
    vlm_vis     = saved / "vlm_vis"
    return {
        "saved":       saved,
        "color_masks": color_masks,
        "vlm_results": vlm_results,
        "vlm_vis":     vlm_vis,
    }


def step1_describe(paths: dict, client, model: str):
    print("\n" + "=" * 60)
    print(f" [{paths['color_masks']}]")
    print("=" * 60)
    from vlm_describe_masks import run_describe
    run_describe(
        color_masks_dir=paths["color_masks"],
        output_path=paths["vlm_results"],
        client=client,
        model=model,
    )


def main():
    parser = argparse.ArgumentParser(description="VLM")
    parser.add_argument("--dataset",    default="causal-lerf", help="lerf-ovs、Ref-lerf、causal-lerf")
    parser.add_argument("--scene_name", default="ramen",  help="figurines、ramen")
    from vlm_client import add_vlm_arguments, make_vlm_client

    add_vlm_arguments(parser)
    parser.add_argument("--skip_describe", action="store_true", help="")
    args = parser.parse_args()

    try:
        import openai  
    except ImportError:
        sys.exit(1)

    paths = get_paths(args.dataset, args.scene_name)
    client, model = make_vlm_client(
        args.api_key,
        args.base_url,
        args.model,
        wait=args.wait_vlm,
        wait_timeout_s=args.wait_vlm_timeout,
    )

    if not args.skip_describe:
        step1_describe(paths, client, model)
    else:
        print(f"\n{paths['vlm_results']}")

if __name__ == "__main__":
    main()
