from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class Config:
    # %% dataset gpt_config
    tokenized_dataset_dir: str = 'datasets/tokenized/SlimPajama-627B'
    validation_dataset_items: int = 12
    test_dataset_items: int = 12
    shuffle: bool = True
    dtype: torch.dtype = torch.float16

    # %% model gpt_config
    full_ckpt_dir: str = 'models/gpt-neo-125m'
    full_ckpt_name: str = 'pytorch_model.bin'
    max_seq_len: int = 1024
    block_size: int = max_seq_len - 1

    # %% training gpt_config
    accum_step_patience: int = 50  # accumulation steps TODO: for debugging only, must up to 1000 for full training
    unbinary_ratio_threshold: float = 0.005  # training termination threshold
    batch_size: int = 8
    accum_batches: int = 8
    betas: Tuple[float, float] = (0.9, 0.95)
    base_lr: float = 1e-4
    base_step_size: int = 100
    base_gamma: float = 0.9
    save_dir: str = 'models/checkpoints'
    file_prefix: str = 'binary-gpt-neo'
    result_dir: str = 'results/'

    # %% binary KD training gpt_config
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
    epoch_patience = 5  # patience for pushing kk
