import os

import torch
import torch.distributed as dist


def log_loss_file(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(f"{data}\n")
    else:
        with open(file_path, 'w') as f:
            f.write(file_path.split(".")[0] + "\n")
            f.write(f"{data}\n")


def batch_early_stop_check(local_early_stopping):
    stop_flag = torch.tensor(int(local_early_stopping), device='cpu')

    dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)

    return stop_flag.item() > 0
