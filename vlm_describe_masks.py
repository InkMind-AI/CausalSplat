
import base64
import json
import os
from pathlib import Path
from typing import Optional
from object_image_utils import encode_object_image

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
MIN_IMAGE_COUNT = 8


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mapping = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    return mapping.get(ext, "image/png")


def describe_folder(client, model: str, folder: Path) -> Optional[str]:
    image_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if len(image_files) < MIN_IMAGE_COUNT:
        print(f"  [SKIP] {folder.name} - image count {len(image_files)} < {MIN_IMAGE_COUNT}")
        return None

    max_images = 10
    if len(image_files) > max_images:
        step = len(image_files) // max_images
        image_files = image_files[::step][:max_images]

    if not image_files:
        print(f"  [SKIP] {folder.name} - no image files found")
        return None

    print(f"  Processing {folder.name}: {len(image_files)} images")

    content = [
        {
            "type": "text",
            "text": (
                "You are given multiple images of the SAME physical 3D object captured from different viewpoints.\n"
                "The foreground object is shown in its TRUE, ORIGINAL colors.\n"
                "The background is pure white - ignore it entirely.\n\n"
                "STEP 1 - Cross-view verification:\n"
                "Look at ALL images together. The object's 3D shape, color, and structure must be consistent "
                "across views. If something appears in only one view and not others, it is likely background noise - discard it.\n\n"
                "STEP 2 - Identify the object:\n"
                "Based on the consistent visual evidence across ALL views, determine what the foreground object is.\n"
                "- Be specific: use precise nouns.\n"
                "- Do NOT guess based on color alone - shape and structure are more reliable.\n\n"
                "Output format (fill in each field exactly; do NOT add extra text outside these fields):\n"
                "1) Category: <specific concrete noun>\n"
                "2) Appearance: <color(s), material, shape, texture, key distinguishing features>\n"
                "3) Affordance: <likely use or function, one short phrase>\n"
                "4) Alternative names: <comma-separated aliases, nicknames, or search terms>\n\n"
                "Be concise and accurate. Only state what is clearly and consistently visible across the images."
            ),
        }
    ]

    for img_path in image_files:
        image_bytes, crop_info = encode_object_image(img_path, extension='.png')
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        mime = 'image/png'
        print(f"  [crop] {img_path.name}: {crop_info['original_size']} -> {crop_info['output_size']} ({crop_info['status']})")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        timeout=float(os.environ.get("VLM_REQUEST_TIMEOUT_S", "120")),
    )
    return response.choices[0].message.content.strip()


def run_describe(color_masks_dir: Path, output_path: Path, client, model: str):
    print("Describing:", color_masks_dir)
    if not color_masks_dir.exists():
        print(f"[ERROR] color_masks directory does not exist: {color_masks_dir}")
        return

    subfolders = sorted(p for p in color_masks_dir.iterdir() if p.is_dir())
    print(f"Found {len(subfolders)} subfolders. Starting...\n")

    results = {}
    for folder in subfolders:
        print(f"[Folder] {folder.name}")
        try:
            desc = describe_folder(client, model, folder)
        except Exception as e:
            desc = f"[ERROR] {e}"
            print(f"  [FAILED] {e}")
        results[folder.name] = desc
        if desc:
            print(f"  Description: {desc[:200]}{'...' if len(desc) > 200 else ''}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {output_path}")
