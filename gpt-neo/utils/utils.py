import os
from typing import List, Optional

import torch
import torch.distributed as dist
from BinaryGPT import BinaryGPTNeoForCausalLM
from transformers import GPTNeoForCausalLM, GPTNeoConfig


def log_loss_file(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(f"{data}\n")
    else:
        with open(file_path, 'w') as f:
            f.write(file_path.split(".")[0] + "\n")
            f.write(f"{data}\n")


def batch_early_stop_check(local_early_stopping):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    stop_flag = torch.tensor(int(local_early_stopping), device=device)

    dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)

    return stop_flag.item() > 0


def get_model_kk(model, ):
    kk_list = []
    aa_list = []
    for _, module in model.named_modules():
        try:
            kk_list.append(module.kk)
            aa_list.append(module.aa)
        except Exception:
            pass
    model_kk = torch.stack(kk_list, dim=0).mean(dim=0).item()
    model_aa = torch.stack(aa_list, dim=0).mean(dim=0).item()
    return model_kk, model_aa


# %% record training states
class TrainingState:
    def __init__(
            self,
            gpt_config: GPTNeoConfig = None,
            binary_model: BinaryGPTNeoForCausalLM = None,
            full_model: GPTNeoForCausalLM = None,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR,
            kk: float = 1.0,
            aa: float = 1.0,
            epoch: int = 0,
            stage: int = 1,
            accum_step: int = 0,
            unbinary_ratio: Optional[List[float]] = None,
            wandb_run_id: Optional[str] = None,
            best_epoch: int = 0,
            epoch_wait: int = 0,
            best_valid_loss: float = float('inf'),
            current_valid_loss: float = float('inf'),
            temp_unbinary_ratio: Optional[List[float]] = None
    ):
        super().__init__()
        # Model state
        self.gpt_config = gpt_config
        self.binary_model = binary_model
        self.full_model = full_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Train steps and settings
        self.kk = kk
        self.aa = aa
        self.epoch = epoch
        self.stage = stage
        self.accum_step = accum_step
        self.unbinary_ratio = unbinary_ratio if unbinary_ratio is not None else []
        self.wandb_run_id = wandb_run_id

        # Temporary parameters
        self.best_epoch = best_epoch
        self.epoch_wait = epoch_wait
        self.best_valid_loss = best_valid_loss
        self.current_valid_loss = current_valid_loss
        self.temp_unbinary_ratio = temp_unbinary_ratio if temp_unbinary_ratio is not None else []

    def state_dict(self):
        """Convert the training state to a dictionary for saving."""
        return {
            'gpt_config': self.gpt_config,
            'binary_model': self.binary_model,
            'full_model': self.full_model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'kk': self.kk,
            'aa': self.aa,
            'epoch': self.epoch,
            "stage": self.stage,
            'accum_step': self.accum_step,
            'unbinary_ratio': self.unbinary_ratio,
            'wandb_run_id': self.wandb_run_id,
            'best_epoch': self.best_epoch,
            'epoch_wait': self.epoch_wait,
            'best_valid_loss': self.best_valid_loss,
            'current_valid_loss': self.current_valid_loss,
            'temp_unbinary_ratio': self.temp_unbinary_ratio
        }

    def save(self, file_path: str):
        """Save the training state to a file."""
        torch.save({'train_state': self.state_dict()}, file_path)

    @classmethod
    def from_dict(cls, state_dict):
        """Create a TrainingState object from a dictionary."""
        # Create a new TrainingState instance
        obj = cls(
            gpt_config=state_dict.get('gpt_config'),
            binary_model=state_dict.get('binary_model'),
            full_model=state_dict.get('full_model'),
            optimizer=state_dict.get('optimizer'),
            lr_scheduler=state_dict.get('lr_scheduler'),
            kk=state_dict.get('kk', 1.0),
            aa=state_dict.get('aa', 1.0),
            epoch=state_dict.get('epoch', 0),
            stage=state_dict.get('stage', 1),
            accum_step=state_dict.get('accum_step', 0),
            unbinary_ratio=state_dict.get('unbinary_ratio', []),
            wandb_run_id=state_dict.get('wandb_run_id'),
            best_epoch=state_dict.get('best_epoch', 0),
            epoch_wait=state_dict.get('epoch_wait', 0),
            best_valid_loss=state_dict.get('best_valid_loss', float('inf')),
            current_valid_loss=state_dict.get('current_valid_loss', float('inf')),
            temp_unbinary_ratio=state_dict.get('temp_unbinary_ratio', [])
        )

        return obj
