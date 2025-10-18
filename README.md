# Instruction to run locally

```shell
docker compose up   
```

## TO RUN ON TPU

```shell
gcloud services enable compute.googleapis.com tpu.googleapis.com

# set your defaults once
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID=$(gcloud config get project)
export ZONE=us-central1-b        # pick a TPU-capable zone
export TPU_NAME=my-tpu             # any name you like
export ACCELERATOR_TYPE=v3-8
export RUNTIME_VERSION=tpu-vm-pt-2.0

# (optional) see available TPU runtimes for your zone
gcloud compute tpus versions list --zone=$ZONE


gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --version=${RUNTIME_VERSION}

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all

gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command '
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y build-essential curl git zlib1g-dev libssl-dev \
  libbz2-dev libreadline-dev libsqlite3-dev llvm libncursesw5-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# pyenv for Python 3.12
if [ ! -d "$HOME/.pyenv" ]; then curl https://pyenv.run | bash; fi
export PYENV_ROOT="$HOME/.pyenv"; export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"; eval "$(pyenv virtualenv-init -)"
pyenv install -s 3.12.5
pyenv virtualenv -f 3.12.5 py312
pyenv activate py312
python -V
pip install -U pip wheel setuptools
# Install matching torch/XLA (2.8) + libtpu
pip install "torch==2.8.*" "torch_xla==2.8.*" "torchvision" \
  -f https://storage.googleapis.com/libtpu-releases/index.html
python - <<PY
import torch, torch_xla; from torch_xla.core import xla_model as xm
print("torch:", torch.__version__, "xla:", torch_xla.__version__)
print("XLA devices:", xm.get_xla_supported_devices())
PY
'

#gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command \
#"sudo apt-get update -y && sudo apt-get install -y libopenblas-dev && \
# pip install -U pip && \
# pip install numpy && \
# pip install 'torch~=2.5.0' 'torch_xla[tpu]~=2.5.0' torchvision \
#     -f https://storage.googleapis.com/libtpu-releases/index.html"

# clone your code and install deps
#gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command \
#"git clone https://github.com/MarengoAlberto/tpu_sharded_training.git ~/proj && \
# cd ~/proj && pyenv activate py312 && pip install -r requirements.txt"
 
 gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command '
set -e
PY=~/.pyenv/versions/py312/bin/python
PIP=~/.pyenv/versions/py312/bin/pip

git clone https://github.com/MarengoAlberto/tpu_sharded_training.git ~/proj
cd ~/proj 
$PIP install -r requirements.txt
'

 gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command '
set -e
PY=~/.pyenv/versions/py312/bin/python
PIP=~/.pyenv/versions/py312/bin/pip
cd ~/proj 
git pull
'
 
# gcloud compute tpus tpu-vm ssh $TPU_NAME \
#  --zone=$ZONE --worker=all --project=$PROJECT_ID --command \
#"cd ~/proj && pyenv activate py312 && PJRT_DEVICE=TPU XLA_USE_BF16=1 python -m main"

# Check the right versions of torch and torch_xla are installed
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command '
set -e
PY="$HOME/.pyenv/versions/py312/bin/python"

echo "== Clean =="
$PY -m pip uninstall -y torch torch-xla libtpu torchvision torchaudio torchtext || true

echo "== Install matching Torch/XLA 2.8 and let XLA pull libtpu =="
$PY -m pip install -U pip wheel setuptools
$PY -m pip install --no-cache-dir \
  "torch==2.8.*" "torch_xla[tpu]==2.8.*" "torchvision" \
  -f https://storage.googleapis.com/libtpu-releases/index.html

echo "== Verify =="
PJRT_DEVICE=TPU $PY - << "PY"
import inspect, torch, torch_xla
from torch_xla.core import xla_model as xm
import pkg_resources
print("torch      :", torch.__version__)
print("torch_xla  :", torch_xla.__version__, "at", inspect.getfile(torch_xla))
print("libtpu ver :", next((str(d.version) for d in pkg_resources.working_set if d.project_name=="libtpu"), "not found"))
print("XLA devices:", xm.get_xla_supported_devices())
PY
'

# RUN your training script
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command '
set -e
PY=~/.pyenv/versions/py312/bin/python
$PY -V
cd ~/proj
PJRT_DEVICE=TPU PYTHONUNBUFFERED=1 XLA_USE_BF16=1 \
  $PY -m main
'
```