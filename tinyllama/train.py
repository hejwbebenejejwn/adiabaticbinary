import os
import platform
import re
import time
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import GPUtil
import fire
import psutil
import torch
import torch.distributed as dist
import wandb
from datasets import TokenizedDataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama import LlamaConfig

from BinaryTinyLlama import BinaryLlamaForCausalLM
from config import Config
from utils import batch_early_stop_check

torch.set_float32_matmul_precision('medium')

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.device(f"cuda:{local_rank}")


# %% record training states
class TrainingState:
    # model state
    llama_config: LlamaConfig
    binary_model: BinaryLlamaForCausalLM

    optimizer: torch.optim.Adam
    lr_scheduler: torch.optim.lr_scheduler.StepLR

    # train steps
    kk: float = 1.
    aa: float = 1.
    epoch: int = 0
    accum_step: int = 0  # Number of gradient accumulation steps
    unbinary_ratio: List[float] = []

    # temporary parameters
    best_epoch: int = 0
    best_valid_loss: float = torch.tensor(float('inf')).item()
    temp_valid_loss_list: List[float] = []
    temp_unbinary_ratio: List[float] = []


# %% trainer setup
def set_up(config: Config) -> None:
    """Build model and Initialize optimizer & learning rate scheduler"""
    llama_config = LlamaConfig.from_pretrained(Config.full_ckpt_dir)
    llama_config.gradient_checkpointing = config.gradient_checkpointing
    llama_config.use_cache = config.use_cache

    ckpt_state_dict = torch.load(Path(Config.full_ckpt_dir) / Config.full_ckpt_name)
    # build binary model
    print(f"Initializing Binary Model...")
    binary_model = BinaryLlamaForCausalLM(llama_config)
    for key, value in binary_model.state_dict().items():
        if key.endswith('.kk') or key.endswith('.aa') or key.endswith('.inv_freq'):
            continue
        elif key in ckpt_state_dict.keys():
            binary_model.state_dict()[key].copy_(value)
        else:
            raise KeyError(f"{key} not found in pretrained model.")
    binary_model.set_kk_stage1(
        torch.tensor([config.init_kk, config.init_aa], dtype=config.dtype, device=binary_model.device))

    optimizer = torch.optim.Adam(binary_model.parameters(), betas=config.betas, lr=config.base_lr)
    # change base_step_size & base_gamma as current lr changes
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.base_step_size, gamma=config.base_gamma)

    TrainingState.binary_model = binary_model
    TrainingState.llama_config = llama_config
    TrainingState.optimizer = optimizer
    TrainingState.lr_scheduler = lr_scheduler


# %% training
def training_step(batch, config: Config) -> torch.Tensor:
    """Training Step"""
    binary_model = TrainingState.binary_model

    input_tokens = batch[:, 0:config.block_size].contiguous()
    target_tokens = batch[:, 1:config.block_size + 1].contiguous().long()

    binary_model_outputs = binary_model.forward(input_tokens, labels=target_tokens)
    loss = binary_model_outputs.loss

    return loss


def training(train_dataloader: DataLoader, config: Config) -> None:
    """Training Epoch"""
    optimizer = TrainingState.optimizer
    lr_scheduler = TrainingState.lr_scheduler
    binary_model = TrainingState.binary_model
    binary_model.train()
    scaler = GradScaler()

    # training loops
    if local_rank == 0:
        kk = TrainingState.kk
        aa = TrainingState.aa
        print(f"[GPU{local_rank}]",
              f"Epoch {TrainingState.epoch},",
              f"kk = {kk:.2f},",
              f"aa = {aa:.2f}",
              f"Training",
              "=" * 10,
              flush=True
              )

    train_epoch_loss = []
    accumulated_loss = 0.0  # initialize for batch accumulation
    batch_wait = 0  # initialize batch_patience
    min_loss = float('inf')  # initialize min_loss
    for i, data in enumerate(train_dataloader):
        start = time.time()
        data = data.to(device, non_blocking=True)

        with autocast(dtype=torch.float16):
            train_loss = training_step(data, config)
            accumulated_loss += train_loss

        scaler.scale(train_loss).backward()

        if (i + 1) % config.accum_batches == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            TrainingState.accum_step += 1
            n_accum = TrainingState.accum_step

            if local_rank == 0:
                elapsed = time.time() - start
                print(f"Epoch Step: {i + 1:6d} | Accumulation Step: {n_accum:3d} | Loss: {train_loss:6.2f}",
                      f"| Tokens/Sec: {config.max_seq_len / elapsed:7.1f}",
                      f"| Learning Rate: {lr_scheduler.get_last_lr()[0]:6.1e}"
                      )

            train_epoch_loss.append(train_loss)
            # Update min_loss if current train_loss is lower
            if train_loss < min_loss:
                min_loss = train_loss
                batch_wait = 0  # reset patience counter if new min_loss found
            else:
                batch_wait += 1
            print({f"min_loss: {min_loss:6.2f}, accumulate step patience: {batch_wait}/{config.accum_step_patience}"})
            # Check if we need to stop early
            dist.barrier()
            local_stop_flag = batch_wait >= config.accum_step_patience
            global_stop_flag = batch_early_stop_check(local_stop_flag)
            if global_stop_flag:
                print("Early stopping triggered.")
                break

    GPUtil.showUtilization()

    # step lr scheduler every epoch
    lr_scheduler.step()
    TrainingState.epoch += 1

    mean_train_loss = torch.mean(torch.tensor(train_epoch_loss)).item()
    wandb.log({'train_loss': mean_train_loss})


# %% validation
def validation_step(batch, config: Config = Config()) -> torch.Tensor:
    """Validation Step"""
    input_tokens = batch[:, 0:config.block_size].contiguous()
    binary_model = TrainingState.binary_model
    valid_loss = binary_model.forward(input_tokens).loss
    return valid_loss


def validation(valid_dataloader: DataLoader, config: Config = Config()) -> None:
    """Validation Epoch"""
    binary_model = TrainingState.binary_model
    binary_model.eval()

    kk = TrainingState.kk
    aa = TrainingState.aa
    if local_rank == 0:
        print(f"[GPU{local_rank}] Epoch {TrainingState.epoch}, kk = {kk:.2f}, aa = {aa:.2f} Validation "
              + "=" * 10, flush=True)
    with torch.no_grad(), autocast(dtype=torch.float16):
        valid_epoch_loss = []
        for data in valid_dataloader:
            data = data.to(device, non_blocking=True)
            valid_loss = validation_step(data, config)
            valid_epoch_loss.append(valid_loss)

    mean_valid_loss = torch.mean(torch.tensor(valid_epoch_loss)).item()
    TrainingState.temp_valid_loss_list.append(mean_valid_loss)
    wandb.log({'valid_loss': mean_valid_loss})

    # calculate unbinary ratio
    unbinary_ratio = []
    for name, param in binary_model.named_parameters():
        if name.endswith('.weight') and name != 'tok_embeddings.weight':
            weight = param.flatten()
            k = binary_model.state_dict()[name[:-7] + '.kk']
            a = binary_model.state_dict()[name[:-7] + '.aa']
            weight_binary = a * torch.tanh(k * (weight - weight.mean())).abs()
            unbinary_ratio.append(weight_binary[weight_binary < 0.99].shape[0] / weight.shape[0])
    max_unbinary_ratio = torch.max(torch.tensor(unbinary_ratio)).item()
    TrainingState.temp_unbinary_ratio.append(max_unbinary_ratio)
    TrainingState.unbinary_ratio.append(max_unbinary_ratio)
    wandb.log({'unbinary_ratio': max_unbinary_ratio})

    if local_rank == 0:
        print(f"Remaining Unbinary Weight: {max_unbinary_ratio * 100:.2f} % ")


# %% early stop methods
def kk_callback(config: Config = Config()) -> None:
    """kk Callback Function:
    Call back at the end of each epoch.
    Push kk and rewind model state."""
    current_epoch = TrainingState.epoch

    # adjust gamma
    mean_unbinary_ratio = torch.mean(torch.tensor(TrainingState.temp_unbinary_ratio)).item()
    last_unbinary_ratio = torch.mean(torch.tensor(TrainingState.temp_unbinary_ratio[-config.patience:])).item()
    if (mean_unbinary_ratio - last_unbinary_ratio) > 0.01:
        config.base_gamma -= 0.1
    elif (mean_unbinary_ratio - last_unbinary_ratio) < 0.01:
        config.base_gamma += 0.1
    TrainingState.lr_scheduler.gamma = config.base_gamma

    # save checkpoint
    file_path = f'{config.save_dir}/{config.file_prefix}-{current_epoch}.pt'
    save_dict = {'train_state': TrainingState}
    torch.save(save_dict, file_path)

    mean_valid_loss = torch.mean(torch.tensor(TrainingState.temp_valid_loss_list)).item()
    last_valid_loss = torch.mean(torch.tensor(TrainingState.temp_valid_loss_list[-config.patience:])).item()

    # update the best validation loss and train epoch
    current_valid_loss = TrainingState.temp_valid_loss_list[-1]
    if current_valid_loss < TrainingState.best_valid_loss:
        TrainingState.best_valid_loss = current_valid_loss
        TrainingState.best_epoch = current_epoch

    if mean_valid_loss < last_valid_loss:
        # remove extra files
        file_to_del = [f'{config.save_dir}/{config.file_prefix}-{i}.pt'
                       for i in range(TrainingState.best_epoch + 1, current_epoch + 1)]
        for file in file_to_del:
            os.remove(file)
        # load best state
        best_file = f'{config.save_dir}/{config.file_prefix}-{TrainingState.best_epoch}.pt'
        checkpoint = torch.load(best_file)['train_state']
        # %% new state
        for attr in TrainingState.__dict__.keys():
            setattr(TrainingState, attr, checkpoint[attr])
        kk = TrainingState.binary_model.state_dict()['model.layers.0.self_attn.q_proj.kk']

        # push kk and aa
        if kk < 1:
            kk += config.kk_lr1
            aa = 1 / kk
        else:
            kk *= config.kk_lr2
            aa = torch.tensor([1.], dtype=kk.dtype, device=device)

        # stage 1
        if kk < config.kk_threshold:
            TrainingState.binary_model.set_kk_stage1(torch.tensor([kk, aa], dtype=kk.dtype, device=device))
        # stage 2
        else:
            TrainingState.binary_model.set_kk_stage2(torch.tensor(config.ratio, dtype=kk.dtype, device=device))

        # record changes
        kk = TrainingState.binary_model.state_dict()['model.layers.0.self_attn.q_proj.kk']
        aa = TrainingState.binary_model.state_dict()['model.layers.0.self_attn.q_proj.aa']
        TrainingState.kk = kk
        TrainingState.aa = aa
        wandb.log({'kk': kk})
        wandb.log({'aa': aa})

        # initialize temporary parameters
        TrainingState.best_epoch = 0
        TrainingState.best_valid_loss = torch.tensor(float('inf')).item()
        TrainingState.temp_valid_loss_list = []
        TrainingState.temp_unbinary_ratio = []


# %% main function
def main(config: Config) -> None:
    # initialize
    wandb.init(project='binary-llama', mode="offline")
    wandb.config.update(config)
    os.makedirs(config.save_dir, exist_ok=True)

    # show platform information
    if local_rank == 0:
        print("=" * 20, "Platform Information", "=" * 20)
        cpu_model = platform.processor()
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024 ** 3)
        memory_usage_percent = memory_info.percent
        cpu_threads = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=1)
        print("-" * 20, "CPU", "-" * 20)
        print(f"CPU: {cpu_model}")
        print(f"Memory: {total_memory_gb:.2f} GB")
        print(f"Memory usage: {memory_usage_percent:.2f}%")
        print(f"CPU threads: {cpu_threads}")
        print(f"CPU usage: {cpu_usage}%")
        gpus = GPUtil.getGPUs()
        i = 0
        print("-" * 20, "GPU", "-" * 20)
        for gpu in gpus:
            print(f"GPU{i}: {gpu.name}")
            print(f"Memory Total: {gpu.memoryTotal / 1024:.2f} GB")
            i += 1
        GPUtil.showUtilization()

    # prepare model
    if local_rank == 0:
        print("=" * 20, "Preparing Model", "=" * 20)
    set_up(config)
    llama_config = TrainingState.llama_config

    # check if resume
    ckpt_files = glob(f'{config.save_dir}/*.pt', recursive=True)
    latest_checkpoint = max(ckpt_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)), default=None)
    if latest_checkpoint:
        if local_rank == 0:
            print("=" * 20, f"Resume from {latest_checkpoint}", "=" * 20)
        checkpoint = torch.load(latest_checkpoint)['train_state']
        for attr in TrainingState.__dict__.keys():
            setattr(TrainingState, attr, checkpoint[attr])

    # prepare datasets
    train_dataset = TokenizedDataset(tokenized_dataset_path=config.tokenized_dataset_dir,
                                     dataset_ratio=config.dataset_ratio, mode='training', shuffle=config.shuffle,
                                     max_seq_len=config.max_seq_len, pad_id=llama_config.pad_token_id)
    valid_dataset = TokenizedDataset(tokenized_dataset_path=config.tokenized_dataset_dir,
                                     dataset_ratio=config.dataset_ratio, mode='validation', shuffle=config.shuffle,
                                     max_seq_len=config.max_seq_len, pad_id=llama_config.pad_token_id)
    test_dataset = TokenizedDataset(tokenized_dataset_path=config.tokenized_dataset_dir,
                                    dataset_ratio=config.dataset_ratio, mode='test', shuffle=config.shuffle,
                                    max_seq_len=config.max_seq_len, pad_id=llama_config.pad_token_id)
    # dataloaders
    dist.init_process_group(backend='nccl')
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  sampler=train_sampler, num_workers=cpu_count(), pin_memory=True)
    valid_sampler = DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size,
                                  sampler=valid_sampler, num_workers=cpu_count(), pin_memory=True)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 sampler=test_sampler, num_workers=cpu_count(), pin_memory=True)

    # start training
    current_unbinary_ratio = 1.
    binary_model = TrainingState.binary_model
    binary_model.to(device)

    # make ddp model
    TrainingState.binary_model = DDP(binary_model, device_ids=[local_rank], output_device=local_rank)

    # run training
    while current_unbinary_ratio > config.unbinary_ratio_threshold:
        if local_rank == 0:
            kk = TrainingState.kk
            if kk < config.kk_threshold:
                print("=" * 20, "Training Stage 1", "=" * 20)
            else:
                print("=" * 20, "Training Stage 2", "=" * 20)

        training(train_dataloader, config)
        validation(valid_dataloader, config)
        kk_callback(config)
        current_unbinary_ratio = TrainingState.unbinary_ratio[-1]


if __name__ == '__main__':
    fire.Fire(main(Config()))
