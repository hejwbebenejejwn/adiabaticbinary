## Binary KD Training

### STEP1

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
cd path/to/project/llama3
pip install -r requirements.txt
```

### STEP2

* Download Data Sets:
```bash
cd path/to/project/llama3/datasets
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```
* Tokenize Data Sets:
```bash
cd path/to/project/llama3/datasets
python prepare_slimpajama.py
```

### STEP3

* Download Pre-trained Model Weights:
```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
```
* Add Parameters to Initial Model:
```bash
cd path/to/project/llama3
python add_parameters.py
```

### STEP4
### Start Training

* single node with multi GPUs:
```bash
cd path/to/project/llama3
torchrun --nproc_per_node=1 --nnodes=8 --node_rank=0 --master_addr="localhost" --master_port=12345 train.py
```

* multi nodes, each with single GPU:
```bash
cd path/to/project/llama3
# on node 1
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="192.168.0.1" --master_port=12345 train.py
# on node 2
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=1 --master_addr="192.168.0.1" --master_port=12345 train.py
# on node 3
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=2 --master_addr="192.168.0.1" --master_port=12345 train.py
# on node 4
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=3 --master_addr="192.168.0.1" --master_port=12345 train.py
```
