#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/minjae/DR-Classification}"
CONDA=/opt/miniforge3/bin/conda
mkdir -p "$ROOT/outputs/logs" "$ROOT/outputs/status"
for ENV in huang_env drgcnn_env dodr_env aordr_env; do
  "$CONDA" run -n "$ENV" python -m pip install --upgrade pip
  "$CONDA" run -n "$ENV" python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
  "$CONDA" run -n "$ENV" python -m pip install numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 pandas==2.3.1 pillow==11.0.0 tqdm==4.67.1 timm==1.0.22 transformers==4.57.1 accelerate==1.9.0 einops==0.8.1 openpyxl==3.1.5 matplotlib==3.10.3 seaborn==0.13.2
  "$CONDA" run -n "$ENV" python -c 'import torch, torchvision, timm; assert torch.cuda.is_available(); print("ENV_READY", torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.get_device_name(0), flush=True)'
done
mkdir -p "$ROOT/google/vit-base-patch16-224"
"$CONDA" run -n dodr_env python -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/vit-base-patch16-224', 'pytorch_model.bin', local_dir='$ROOT/google/vit-base-patch16-224')"
touch "$ROOT/outputs/status/environment.done"
echo "FINISHED server environment setup"
