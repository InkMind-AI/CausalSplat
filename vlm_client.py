

from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request
from typing import Tuple
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen3-vl-30b-a3b-instruct"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_CLOUD_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LOCAL_API_KEY_PLACEHOLDER = "EMPTY"


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/") if url else url


def is_local_vlm_base_url(url: str) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "0.0.0.0")


def resolve_vlm_config(
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Tuple[str, str, str]:
    model_name = model or os.environ.get("VLM_MODEL_NAME", "") or DEFAULT_MODEL

    url = base_url or os.environ.get("VLM_BASE_URL", "")
    if not url:
        url = DEFAULT_CLOUD_BASE_URL
    url = _normalize_base_url(url)

    key = api_key
    if not key:
        key = (
            os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )

    if not key and is_local_vlm_base_url(url):
        key = LOCAL_API_KEY_PLACEHOLDER

    return key, url, model_name


def wait_for_vlm(
    base_url: str,
    timeout_s: float = 180,
    interval_s: float = 5.0,
) -> None:
    models_url = _normalize_base_url(base_url) + "/models"
    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    print(f"[VLM] ready: {models_url}")
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)
        time.sleep(interval_s)
    raise SystemExit(
        f"[VLM] server not ready after {timeout_s}s: {models_url} ({last_err})"
    )


def make_vlm_client(
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    *,
    wait: bool = False,
    wait_timeout_s: float = 180,
):
    from openai import OpenAI

    key, url, model_name = resolve_vlm_config(
        api_key, base_url, model
    )
    if not url:
        raise SystemExit(
            "[VLM] Missing base_url. Use --base_url or set VLM_BASE_URL."
        )
    if not key:
        raise SystemExit(
            "[VLM] Missing API key. Pass --api_key, set DASHSCOPE_API_KEY, "
            "or set OPENAI_API_KEY. Local endpoints accept EMPTY."
        )
    if wait:
        wait_for_vlm(url, timeout_s=wait_timeout_s)
    print(f"[VLM] base_url={url}, model={model_name}")
    return OpenAI(api_key=key, base_url=url), model_name


def add_vlm_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_wait: bool = True,
) -> None:
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key (or DASHSCOPE_API_KEY / OPENAI_API_KEY). "
        "Local vLLM: omit or use EMPTY.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="OpenAI-compatible endpoint for an already running model service. "
        "Override with VLM_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"VLM model id (default: {DEFAULT_MODEL}, or VLM_MODEL_NAME).",
    )
    if include_wait:
        parser.add_argument(
            "--wait_vlm",
            action="store_true",
            help="Wait until the configured service responds to GET /v1/models.",
        )
        parser.add_argument(
            "--wait_vlm_timeout",
            type=float,
            default=180,
            help="Seconds to wait when --wait_vlm is set.",
        )
