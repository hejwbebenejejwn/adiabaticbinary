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
