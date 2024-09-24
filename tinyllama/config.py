from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Config:
    # %% dataset config
    tokenized_dataset_dir: str = 'datasets/tokenized/SlimPajama-627B'
    dataset_ratio: list = None
    shuffle: bool = True
    dtype: torch.dtype = torch.float16

    # %% model config
    full_ckpt_dir: str = 'models/TinyLlama-1.1B-intermediate-step-715k-1.5T'
    full_ckpt_name: str = 'pytorch_model.bin'
    tokenizer_path: str = 'models/TinyLlama-1.1B-intermediate-step-715k-1.5T/tokenizer.model'
    llama_config_path: str = 'models/TinyLlama-1.1B-intermediate-step-715k-1.5T/config.json'
    max_seq_len: int = 2048
    block_size: int = max_seq_len - 1

    # %% training config
    gradient_checkpointing: bool = True
    use_cache: bool = False
    unbinary_ratio_threshold: float = 0.005  # training termination threshold
    kk_threshold: float = 100.  # kk threshold to divide training stage 1 and stage 2 (for present)
    batch_size: int = 16
    accum_batches: int = 1
    betas: Tuple[float, float] = (0.9, 0.95)
    base_lr: float = 1e-4
    base_step_size: int = 100
    base_gamma: float = 0.9
    save_dir: str = 'models/BinaryLlama'
    file_prefix: str = 'binary-llama'

    # %% binary KD training config
    kd_temperature: float = 10.
    kd_alpha: float = 0.1
    aa_lr: float = 0.4
    kk_lr: float = 1.25
    ratio: float = 0.1
    patience = 15
