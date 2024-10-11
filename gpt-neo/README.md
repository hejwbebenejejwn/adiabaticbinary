## Binary KD Training

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
cd path/to/project/gpt-neo/models
git lfs clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T
```

* Add Parameters to Initial Model:

```bash
cd path/to/project/tinyllama
python add_parameters.py
```

## STEP3

* Download Data Sets:

```bash
cd path/to/project/tinyllama/datasets
git lfs clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```

* Tokenize Data Sets:

```bash
cd path/to/project/tinyllama/datasets
python prepare_slimpajama.py
```

## STEP4

### Start Training

* single node with multi GPUs:

```bash
cd path/to/project/tinyllama
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
