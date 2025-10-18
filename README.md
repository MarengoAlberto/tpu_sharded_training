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

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command \
"sudo apt-get update -y && sudo apt-get install -y libopenblas-dev && \
 pip install -U pip && \
 pip install numpy && \
 pip install 'torch~=2.5.0' 'torch_xla[tpu]~=2.5.0' torchvision \
     -f https://storage.googleapis.com/libtpu-releases/index.html"

# clone your code and install deps
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command \
"git clone https://github.com/MarengoAlberto/tpu_sharded_training.git ~/proj && \
 cd ~/proj && pip install -r requirements.txt"
```