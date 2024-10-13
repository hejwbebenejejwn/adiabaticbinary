from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class Config:
    # %% dataset config
    dataset_dir: str = 'datasets/General-Stories-Collection'
    tokenized_dataset_dir: str = 'datasets/tokenized/General-Stories-Collection'
    validation_dataset_items: int = 2000
    test_dataset_items: int = 2000
    shuffle: bool = True
    dtype: torch.dtype = torch.float32

    # %% model gpt_config
    full_ckpt_dir: str = 'models/gpt-neo'
    full_ckpt_name: str = 'pytorch_model.bin'
    max_seq_len: int = 1024
    block_size: int = max_seq_len - 1

    # %% training config
    full_epoch_patience = 10  # patience for full model training
    accum_step_patience: int = 100  # accumulation steps TODO: for debugging only, must up to 1000 for full training
    batch_size: int = 32
    accum_batches: int = 8

    unbinary_ratio_threshold: float = 0.005  # training termination threshold
    valid_loss_threshold: float = 0.01  # loss tolerance for current_valid_loss and best_valid_loss
    betas: Tuple[float, float] = (0.9, 0.95)
    base_lr: float = 1e-4
    base_step_size: int = 100
    base_gamma: float = 0.9

    full_save_dir: str = 'models/full_checkpoints'
    full_file_prefix: str = 'gpt-neo'
    binary_save_dir: str = 'models/binary_checkpoints'
    binary_file_prefix: str = 'binary-gpt-neo'

    # %% binary KD training config
    kk_epoch_patience = 5  # patience for pushing kk
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
