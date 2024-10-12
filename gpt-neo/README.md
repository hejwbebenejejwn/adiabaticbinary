## Binary KD Training for GPT-NEO

## STEP1

### Environments Requirements:

* Install git lfs (Ubuntu based Linux)

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

* Install git lfs (Windows)

```bash
git lfs install
```

* Install python packages

```bash
cd path/to/project/gpt-neo
pip install -r requirements.txt
```

## STEP2

* Download Pre-trained Model Weights:

```bash
cd models
git lfs clone https://huggingface.co/EleutherAI/gpt-neo-125m
```

* Copy key model parameters:

```bash
mkdir gpt-neo
cd gpt-neo-125m
cp *.json ../gpt-neo
```

## STEP3

* Download Data Sets:

```bash
cd ../../datasets
git lfs clone https://huggingface.co/datasets/ajibawa-2023/General-Stories-Collection
```

* Tokenize Data Sets:

```bash
cd ..
python datasets/prepare_general_stories.py
```

## STEP4

### Pre-Training

* single node with multi GPUs:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 pretrain.py
```

* multi nodes, each with single GPU:

```bash
cd path/to/project/tinyllama
# on node 1
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="192.168.0.1" --master_port=12345 pretrain.py
# on node 2
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=1 --master_addr="192.168.0.1" --master_port=12345 pretrain.py
# on node 3
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr="192.168.0.1" --master_port=12345 pretrain.py
# on node 4
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=3 --master_addr="192.168.0.1" --master_port=12345 pretrain.py
```

### Start Training

* single node with multi GPUs:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 train.py
```

* multi nodes, each with single GPU:

```bash
cd path/to/project/tinyllama
# on node 1
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="192.168.0.1" --master_port=12345 train.py
# on node 2
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=1 --master_addr="192.168.0.1" --master_port=12345 train.py
# on node 3
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr="192.168.0.1" --master_port=12345 train.py
# on node 4
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=3 --master_addr="192.168.0.1" --master_port=12345 train.py
```
