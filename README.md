# [MM 2026] CausalSplat: Towards Comprehensive Hierarchical Reasoning in 3D Gaussian Splatting

[Project Page](https://inkmind-ai.github.io/CausalSplat/)


## Installation

Requires Linux, an NVIDIA GPU, CUDA-enabled PyTorch, and a compatible CUDA toolkit and C++ compiler.

```bash
git clone https://github.com/InkMind-AI/CausalSplat.git
cd CausalSplat
conda create -n causalsplat python=3.10 -y
conda activate causalsplat

# Select a PyTorch build compatible with your CUDA environment.
python -m pip install torch torchvision
python -m pip install numpy scipy pillow opencv-python tqdm plyfile hdbscan openai huggingface_hub
python -m pip install ./casual-main/third_party/segment-anything-langsplat
python -m pip install --no-build-isolation ./casual-main/submodules/simple-knn
python -m pip install --no-build-isolation ./casual-main/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./casual-main/submodules/diff-gaussian-rasterization_contrastive_f
```

Install PyTorch3D or Open3D separately for optional geometry and visualization paths.

## Data

Supported scenes: `figurines`, `ramen`, `teatime`, and `waldo_kitchen`.

```text
casual-main/data/causal-lerf/<scene>/
├── images/                     # Scene images
├── sparse/0/                   # COLMAP reconstruction
├── json/test_json/             # Evaluation queries
├── mask/                       # Ground-truth masks
├── language_features_test/     # Generated SAM outputs
└── gs/                         # Generated training artifacts
```

Download datasets and the SAM ViT-H checkpoint separately. Logs, data, weights, and generated outputs are excluded from this repository.

## Run

Run from the repository root with the project environment activated. Start your model service separately, then connect through its OpenAI-compatible API and configure a downloaded SAM checkpoint:

```bash
export SAM_CKPT_PATH="$PWD/weights/sam_vit_h_4b8939.pth"
export VLM_BASE_URL="http://127.0.0.1:8001/v1"
export OPENAI_API_KEY="EMPTY"  # Replace for an authenticated service.

python casual-main/run_causal_lerf.py \
  --scene_name ramen \
  --python "$(command -v python)" \
  --sam_ckpt_path "$SAM_CKPT_PATH"
```

The default model identifier is `qwen3-vl-30b-a3b-instruct`. The launcher runs reconstruction, preprocessing, feature training, clustering, description, scene graph construction, and evaluation. Use `--help` for stage-specific skip and rerun options.

## Evaluation

After generating the scene graph and object scores:

```bash
python casual-main/reason_llm.py \
  --dataset causal-lerf --scene_name ramen --level 2
```

Results are saved to `saved_masks/causal-lerf/<scene>/llm_reasoning_results_lvl2.json`, including `overall_miou`.

**We have tuned the parameters in this implementation, so evaluation scores may be higher than those obtained with earlier settings.**

When reporting results, specify the parameters, model, and evaluation protocol.

## Acknowledgments and License

Built with 3D Gaussian Splatting, SAM-based segmentation, and bundled third-party libraries. Existing copyright notices and attributions are retained. Follow the applicable terms for the code, datasets, and model checkpoints.
