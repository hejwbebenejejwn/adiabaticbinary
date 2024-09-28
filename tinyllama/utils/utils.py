import os
from typing import List, Optional

import torch
import torch.distributed as dist
from BinaryTinyLlama import BinaryLlamaForCausalLM
from transformers import LlamaForCausalLM
from transformers.models.llama import LlamaConfig


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
    model_kk = torch.stack(kk_list, dim=0).mean(dim=0)
    model_aa = torch.stack(aa_list, dim=0).mean(dim=0)
    return model_kk, model_aa


# %% record training states
class TrainingState:
    def __init__(
            self,
            llama_config: LlamaConfig = None,
            binary_model: BinaryLlamaForCausalLM = None,
            full_model: LlamaForCausalLM = None,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR,
            kk: float = 1.0,
            aa: float = 1.0,
            epoch: int = 0,
            accum_step: int = 0,
            unbinary_ratio: Optional[List[float]] = None,
            wandb_run_id: Optional[str] = None,
            best_epoch: int = 0,
            best_valid_loss: float = float('inf'),
            temp_valid_loss_list: Optional[List[float]] = None,
            temp_unbinary_ratio: Optional[List[float]] = None
    ):
        super().__init__()
        # Model state
        self.llama_config = llama_config
        self.binary_model = binary_model
        self.full_model = full_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Train steps and settings
        self.kk = kk
        self.aa = aa
        self.epoch = epoch
        self.accum_step = accum_step
        self.unbinary_ratio = unbinary_ratio if unbinary_ratio is not None else []
        self.wandb_run_id = wandb_run_id

        # Temporary parameters
        self.best_epoch = best_epoch
        self.best_valid_loss = best_valid_loss
        self.temp_valid_loss_list = temp_valid_loss_list if temp_valid_loss_list is not None else []
        self.temp_unbinary_ratio = temp_unbinary_ratio if temp_unbinary_ratio is not None else []

    def state_dict(self):
        """Convert the training state to a dictionary for saving."""
        return {
            'llama_config': self.llama_config,
            'binary_model': self.binary_model,
            'full_model': self.full_model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'kk': self.kk,
            'aa': self.aa,
            'epoch': self.epoch,
            'accum_step': self.accum_step,
            'unbinary_ratio': self.unbinary_ratio,
            'wandb_run_id': self.wandb_run_id,
            'best_epoch': self.best_epoch,
            'best_valid_loss': self.best_valid_loss,
            'temp_valid_loss_list': self.temp_valid_loss_list,
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
            llama_config=state_dict.get('llama_config'),
            binary_model=state_dict.get('binary_model'),
            full_model=state_dict.get('full_model'),
            optimizer=state_dict.get('optimizer'),
            lr_scheduler=state_dict.get('lr_scheduler'),
            kk=state_dict.get('kk', 1.0),
            aa=state_dict.get('aa', 1.0),
            epoch=state_dict.get('epoch', 0),
            accum_step=state_dict.get('accum_step', 0),
            unbinary_ratio=state_dict.get('unbinary_ratio', []),
            wandb_run_id=state_dict.get('wandb_run_id'),
            best_epoch=state_dict.get('best_epoch', 0),
            best_valid_loss=state_dict.get('best_valid_loss', float('inf')),
            temp_valid_loss_list=state_dict.get('temp_valid_loss_list', []),
            temp_unbinary_ratio=state_dict.get('temp_unbinary_ratio', [])
        )

        return obj

    # save state_dict only
    # def state_dict(self):
    #     """Convert the training state to a dictionary for saving."""
    #     return {
    #         'llama_config': self.llama_config,
    #         'binary_model_state_dict': self.binary_model.state_dict() if self.binary_model else None,
    #         'full_model_state_dict': self.full_model.state_dict() if self.full_model else None,
    #         'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
    #         'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
    #         'kk': self.kk,
    #         'aa': self.aa,
    #         'epoch': self.epoch,
    #         'accum_step': self.accum_step,
    #         'unbinary_ratio': self.unbinary_ratio,
    #         'wandb_run_id': self.wandb_run_id,
    #         'best_epoch': self.best_epoch,
    #         'best_valid_loss': self.best_valid_loss,
    #         'temp_valid_loss_list': self.temp_valid_loss_list,
    #         'temp_unbinary_ratio': self.temp_unbinary_ratio
    #     }
    #
    # def save(self, file_path: str):
    #     """Save the training state to a file."""
    #     save_dict = {'train_state': self.state_dict()}
    #     torch.save(save_dict, file_path)
    #
    # @classmethod
    # def from_dict(cls, state_dict, binary_model=None, full_model=None, optimizer=None, lr_scheduler=None):
    #     """Create a TrainingState object from a dictionary."""
    #     # Create a new TrainingState instance
    #     obj = cls(
    #         llama_config=state_dict.get('llama_config'),
    #         binary_model=binary_model,
    #         full_model=full_model,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler,
    #         kk=state_dict.get('kk', 1.0),
    #         aa=state_dict.get('aa', 1.0),
    #         epoch=state_dict.get('epoch', 0),
    #         accum_step=state_dict.get('accum_step', 0),
    #         unbinary_ratio=state_dict.get('unbinary_ratio', []),
    #         wandb_run_id=state_dict.get('wandb_run_id'),
    #         best_epoch=state_dict.get('best_epoch', 0),
    #         best_valid_loss=state_dict.get('best_valid_loss', float('inf')),
    #         temp_valid_loss_list=state_dict.get('temp_valid_loss_list', []),
    #         temp_unbinary_ratio=state_dict.get('temp_unbinary_ratio', [])
    #     )
    #
    #     # Load state_dicts into model, optimizer, and lr_scheduler if available
    #     if binary_model and state_dict.get('binary_model_state_dict'):
    #         binary_model.load_state_dict(state_dict['binary_model_state_dict'])
    #     if full_model and state_dict.get('full_model_state_dict'):
    #         full_model.load_state_dict(state_dict['full_model_state_dict'])
    #     if optimizer and state_dict.get('optimizer_state_dict'):
    #         optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    #     if lr_scheduler and state_dict.get('lr_scheduler_state_dict'):
    #         lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
    #
    #     return obj
