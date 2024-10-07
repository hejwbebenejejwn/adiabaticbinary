from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class Config:
    # %% dataset config
    tokenized_dataset_dir: str = 'datasets/tokenized/SlimPajama-627B'
    validation_dataset_items: int = 2000
    test_dataset_items: int = 2000
    shuffle: bool = True
    dtype: torch.dtype = torch.float16

    # %% model config
    full_ckpt_dir: str = 'models/TinyLlama-1.1B-intermediate-step-715k-1.5T'
    full_ckpt_name: str = 'pytorch_model.bin'
    tokenizer_path: str = 'models/TinyLlama-1.1B-intermediate-step-715k-1.5T/tokenizer.model'
    llama_config_path: str = 'models/TinyLlama-1.1B-intermediate-step-715k-1.5T/config.json'
    max_seq_len: int = 1024
    block_size: int = max_seq_len - 1

    # %% training config
    accum_step_patience: int = 50  # accumulation steps TODO: for debugging only, must up to 1000 for full training
    gradient_checkpointing: bool = True
    use_cache: bool = False
    unbinary_ratio_threshold: float = 0.005  # training termination threshold
    batch_size: int = 8
    accum_batches: int = 8
    betas: Tuple[float, float] = (0.9, 0.95)
    base_lr: float = 1e-4
    base_step_size: int = 100
    base_gamma: float = 0.9
    save_dir: str = 'models/checkpoints'
    file_prefix: str = 'binary-llama'
    result_dir: str = 'results/'

    # %% binary KD training config
    loss_fn_mse: F.mse_loss = F.mse_loss
    kd_training: bool = False  # if you use knowledge distillation training
    kd_temperature: float = 10.
    kd_alpha: float = 0.1
    initial_kk: float = 0.5
    initial_aa: float = 1 / initial_kk
    kk_lr1: float = 0.5
    kk_lr2: float = 1.25
    kk_threshold: float = 100.  # kk threshold to divide training stage 1 and stage 2
    # TODO: determined according to initial model weights
    ratio: float = 0.1  # ratio of remaining kk being pushed to binary at each push in stage 2
    patience = 3  # last epoches to calculate mean temporary validation loss
