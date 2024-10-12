import copy
import os
import platform
import re
import time
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path

import GPUtil
import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import TokenizedDataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from BinaryGPT import BinaryGPTNeoForCausalLM
from config import Config
from utils import batch_early_stop_check, get_model_kk, TrainingState

# torch.autograd.set_detect_anomaly(True)

torch.set_float32_matmul_precision('medium')

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.device(f"cuda:{local_rank}")


# %% trainer setup
def set_up(config: Config, train_state: TrainingState) -> TrainingState:
    """Build model and Initialize optimizer & learning rate scheduler"""
    gpt_config = GPTNeoConfig.from_pretrained(Config.full_ckpt_dir)
    train_state.gpt_config = gpt_config

    ckpt_state_dict = torch.load(Path(Config.full_ckpt_dir) / Config.full_ckpt_name, map_location=device)
    # build binary model
    if local_rank == 0:
        print(f"Initializing Binary Model...")
    binary_model = BinaryGPTNeoForCausalLM(gpt_config)
    for key, _ in binary_model.state_dict().items():
        if key.endswith('.kk') or key.endswith('.aa'):
            continue
        elif key in ckpt_state_dict.keys():
            binary_model.state_dict()[key].copy_(ckpt_state_dict[key])
        else:
            raise KeyError(f"{key} not found in pretrained model.")
    binary_model.set_kk_stage1([config.initial_kk, config.initial_aa])
    train_state.binary_model = binary_model
    train_state.kk = config.initial_kk
    train_state.aa = config.initial_aa

    # build gpt-neo model
    if config.kd_training:
        if local_rank == 0:
            print(f"Initializing Full Precision Model...")
        full_model = GPTNeoForCausalLM(gpt_config)
        for key, _ in full_model.state_dict().items():
            if key in ckpt_state_dict.keys():
                full_model.state_dict()[key].copy_(ckpt_state_dict[key])
            else:
                raise KeyError(f"{key} not found in pretrained model.")
        train_state.full_model = full_model

    optimizer = torch.optim.Adam(binary_model.parameters(), betas=config.betas, lr=config.base_lr)
    train_state.optimizer = optimizer
    # change base_step_size & base_gamma as current lr changes
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.base_step_size, gamma=config.base_gamma)
    train_state.lr_scheduler = lr_scheduler

    return train_state


# %% training
def training_step(
        batch,
        config: Config,
        binary_model: BinaryGPTNeoForCausalLM,
        full_model: GPTNeoForCausalLM,
        loss_fn_mse: F.mse_loss,
) -> torch.Tensor:
    """Training Step"""

    input_tokens = batch[:, 0:config.block_size].contiguous()
    target_tokens = batch[:, 1:config.block_size + 1].contiguous().long()

    if config.kd_training:
        kd_temperature = config.kd_temperature

        binary_model_outputs = binary_model.forward(input_tokens, labels=target_tokens)
        binary_logits = binary_model_outputs.logits
        binary_logits = binary_logits.view(-1, binary_logits.size(-1))
        binary_kd_probs = torch.softmax(binary_logits / kd_temperature, dim=-1)
        binary_kd_probs = binary_kd_probs.view(-1, binary_kd_probs.size(-1))

        full_logits = full_model.forward(input_tokens).logits
        full_logits = full_logits.view(-1, binary_logits.size(-1))
        full_kd_probs = torch.softmax(full_logits / kd_temperature, dim=-1)
        full_kd_probs = full_kd_probs.view(-1, full_kd_probs.size(-1))

        # logit KD loss
        loss_ce = binary_model_outputs.loss
        loss_mse = loss_fn_mse(input=binary_kd_probs, target=full_kd_probs)
        loss = loss_ce + loss_mse * config.kd_alpha
    else:
        binary_model_outputs = binary_model.forward(input_tokens, labels=target_tokens)
        loss = binary_model_outputs.loss

    return loss


def training(train_dataloader: DataLoader, config: Config, train_state: TrainingState) -> TrainingState:
    """Training Epoch"""
    binary_model = train_state.binary_model
    binary_model.train()

    full_model = train_state.full_model
    if full_model is not None:
        full_model.eval()

    loss_fn_mse = config.loss_fn_mse

    optimizer = train_state.optimizer
    lr_scheduler = train_state.lr_scheduler
    scaler = GradScaler()

    # Ensure the data is shuffled and redistributed for each epoch
    train_sampler = train_dataloader.sampler
    train_sampler.set_epoch(train_state.epoch)

    # training loops
    if local_rank == 0:
        print(
            f"[GPU{local_rank}]",
            f"Epoch {train_state.epoch},",
            f"kk = {train_state.kk:.2f},",
            f"aa = {train_state.aa:.2f}",
            "Training",
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
            train_loss = training_step(data, config, binary_model, full_model, loss_fn_mse)
            accumulated_loss += train_loss

        scaler.scale(train_loss).backward()

        if (i + 1) % config.accum_batches == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            train_state.accum_step += 1
            n_accum = train_state.accum_step

            if local_rank == 0:
                elapsed = time.time() - start
                print(
                    f"Epoch Step: {i + 1:6d} | Accumulation Step: {n_accum:3d} | Loss: {train_loss:6.2f}",
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

            if local_rank == 0:
                print(f"accumulation step wait: {batch_wait}/{config.accum_step_patience}")

            # Check if we need to stop early
            dist.barrier()
            local_stop_flag = batch_wait >= config.accum_step_patience
            global_stop_flag = batch_early_stop_check(local_stop_flag)
            if global_stop_flag:
                if local_rank == 0:
                    print("batch early stopping triggered.")
                break

    GPUtil.showUtilization()

    # step lr scheduler every epoch
    lr_scheduler.step()

    mean_train_loss = torch.mean(torch.tensor(train_epoch_loss)).item()
    wandb.log({'train_loss': mean_train_loss})

    return train_state


# %% validation
def validation_step(batch, config: Config, binary_model: BinaryGPTNeoForCausalLM) -> torch.Tensor:
    """Validation Step"""
    input_tokens = batch[:, 0:config.block_size].contiguous()
    target_tokens = batch[:, 1:config.block_size + 1].contiguous().long()

    valid_loss = binary_model.forward(input_tokens, labels=target_tokens).loss

    return valid_loss


def validation(valid_dataloader: DataLoader, config: Config, train_state: TrainingState) -> TrainingState:
    """Validation Epoch"""
    binary_model = train_state.binary_model
    binary_model.eval()

    if local_rank == 0:
        print(
            f"[GPU{local_rank}] Epoch {train_state.epoch},",
            f"kk = {train_state.kk:.2f},",
            f"aa = {train_state.aa:.2f} Validation",
            "=" * 10,
            flush=True
        )
    with torch.no_grad(), autocast(dtype=torch.float16):
        valid_epoch_loss = []
        for i, data in enumerate(valid_dataloader):
            data = data.to(device, non_blocking=True)
            valid_loss = validation_step(data, config, binary_model)
            valid_epoch_loss.append(valid_loss)
            if local_rank == 0:
                print(f"Epoch Step: {i + 1:6d} | Loss: {valid_loss:6.2f}")

    mean_valid_loss = torch.mean(torch.tensor(valid_epoch_loss)).item()
    train_state.current_valid_loss = mean_valid_loss
    wandb.log({'valid_loss': mean_valid_loss})

    # calculate unbinary ratio
    unbinary_ratio = []
    for name, module in binary_model.named_modules():
        try:
            weight = module.weight.flatten()
            k = module.kk
            a = module.aa
            weight_binary = a * torch.tanh(k * (weight - weight.mean())).abs()
            unbinary_ratio.append(weight_binary[weight_binary < 0.99].shape[0] / weight.shape[0])
        except Exception:
            pass
    max_unbinary_ratio = torch.max(torch.tensor(unbinary_ratio)).item()
    train_state.temp_unbinary_ratio.append(max_unbinary_ratio)
    train_state.unbinary_ratio.append(max_unbinary_ratio)
    wandb.log({'unbinary_ratio': max_unbinary_ratio})

    if local_rank == 0:
        print(f"Remaining Unbinary Weight: {max_unbinary_ratio * 100:.2f} % ")

    return train_state


# %% early stop methods
def kk_callback(config: Config, train_state: TrainingState) -> TrainingState:
    """kk Callback Function:
    Call back at the end of each epoch.
    Push kk and rewind model state."""
    current_epoch = train_state.epoch

    # adjust gamma
    mean_unbinary_ratio = torch.mean(torch.tensor(train_state.temp_unbinary_ratio)).item()
    last_unbinary_ratio = torch.mean(torch.tensor(train_state.temp_unbinary_ratio[-config.kk_epoch_patience:])).item()
    if (mean_unbinary_ratio - last_unbinary_ratio) > 0.01:
        config.base_gamma -= 0.1
    elif (mean_unbinary_ratio - last_unbinary_ratio) < 0.01:
        config.base_gamma += 0.1
    train_state.lr_scheduler.gamma = config.base_gamma

    if local_rank == 0:
        print(f"Current Valid Loss: {train_state.current_valid_loss:6.5f}")
        print(f"Best Validation Loss: {train_state.best_valid_loss:6.5f}")

    # update the best validation loss and train epoch
    current_valid_loss = train_state.current_valid_loss
    best_valid_loss = train_state.best_valid_loss
    if current_valid_loss <= best_valid_loss:
        train_state.best_valid_loss = current_valid_loss
        train_state.best_epoch = current_epoch
        train_state.epoch_wait = 0
    else:
        train_state.epoch_wait += 1

    if local_rank == 0:
        print(f"Best Epoch: {train_state.best_epoch:1d}")
        print(f"Epoch Wait: {train_state.epoch_wait:1d}/{config.kk_epoch_patience}")

    # save checkpoint
    file_path = f'{config.binary_save_dir}/{config.binary_file_prefix}-{current_epoch}.pt'
    train_state.save(file_path)

    # Check if we need to stop early
    dist.barrier()
    local_stop_flag = train_state.epoch_wait >= config.kk_epoch_patience
    global_stop_flag = batch_early_stop_check(local_stop_flag)
    if global_stop_flag:
        print("Pushing kk ...")
        # remove extra files
        file_to_del = [f'{config.binary_save_dir}/{config.binary_file_prefix}-{i}.pt'
                       for i in range(train_state.best_epoch + 1, current_epoch + 1)]
        for file in file_to_del:
            os.remove(file)
        # load best state
        best_file = f'{config.binary_save_dir}/{config.binary_file_prefix}-{train_state.best_epoch}.pt'
        print(f"Return back to Epoch {train_state.best_epoch}.")
        train_state = TrainingState.from_dict(torch.load(best_file, map_location=device)['train_state'])

        # %% push kk and aa
        kk, _ = get_model_kk(train_state.binary_model)
        if kk < 1:
            kk += config.kk_lr1
            aa = 1 / kk
        else:
            kk *= config.kk_lr2
            aa = 1.

        # train state determination
        if local_rank == 0 and train_state.stage == 1:
            try_binary_model = copy.deepcopy(train_state.binary_model.module).to('cpu')
            try_binary_model.set_kk_stage2(config.ratio)
            try_kk, _ = get_model_kk(try_binary_model)
            if try_kk - kk < config.kk_threshold:
                train_state.stage = 2
            del try_binary_model
            torch.cuda.empty_cache()
        # stage 1
        if train_state.stage == 1:
            train_state.binary_model.module.set_kk_stage1([kk, aa])
        # stage 2
        elif train_state.stage == 2:
            train_state.binary_model.module.set_kk_stage2(config.ratio)
        else:
            raise ValueError(f"Unexpected stage: {train_state.stage}")

        # record changes
        kk, aa = get_model_kk(train_state.binary_model)
        train_state.kk = kk
        train_state.aa = aa
        wandb.log({'kk': kk})
        wandb.log({'aa': aa})

        # initialize temporary parameters
        train_state.best_valid_loss = float('inf')
        train_state.current_valid_loss = best_valid_loss
        train_state.temp_unbinary_ratio = []

    return train_state


# %% main function
def main(config: Config) -> None:
    # initialize
    os.makedirs(config.binary_save_dir, exist_ok=True)
    dist.init_process_group(backend='nccl')

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

    # check if resume
    ckpt_files = glob(f'{config.binary_save_dir}/*.pt', recursive=True)
    latest_checkpoint = max(ckpt_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)), default=None)
    if latest_checkpoint:
        if local_rank == 0:
            print("=" * 20, f"Resume from {latest_checkpoint}", "=" * 20)
            print("Loading Checkpoint...")
        train_state = TrainingState.from_dict(torch.load(latest_checkpoint, map_location=device)['train_state'])
        # initialize
        wandb.init(project='binary-gpt-neo', mode="offline", id=train_state.wandb_run_id)
    else:
        # initialize
        train_state = TrainingState()
        run = wandb.init(project='binary-gpt-neo', mode="offline")
        train_state.wandb_run_id = run.id
        wandb.config.update(config)
        # prepare model
        if local_rank == 0:
            print("=" * 20, "Preparing Model", "=" * 20)
        train_state = set_up(config, train_state)

        # make ddp model
        binary_model = train_state.binary_model.to(device)
        train_state.binary_model = DDP(binary_model, device_ids=[local_rank], output_device=local_rank)
        if config.kd_training:
            full_model = train_state.full_model.to(device)
            train_state.full_model = DDP(full_model, device_ids=[local_rank], output_device=local_rank)
    gpt_neo_config = train_state.gpt_config

    # prepare datasets
    train_dataset = TokenizedDataset(
        tokenized_dataset_path=config.tokenized_dataset_dir,
        validation_dataset_items=config.validation_dataset_items,
        test_dataset_items=config.test_dataset_items,
        mode='training',
        shuffle=config.shuffle,
        max_seq_len=config.max_seq_len,
        pad_id=gpt_neo_config.pad_token_id
    )
    valid_dataset = TokenizedDataset(
        tokenized_dataset_path=config.tokenized_dataset_dir,
        validation_dataset_items=config.validation_dataset_items,
        test_dataset_items=config.test_dataset_items,
        mode='validation',
        shuffle=config.shuffle,
        max_seq_len=config.max_seq_len,
        pad_id=gpt_neo_config.pad_token_id
    )
    test_dataset = TokenizedDataset(
        tokenized_dataset_path=config.tokenized_dataset_dir,
        validation_dataset_items=config.validation_dataset_items,
        test_dataset_items=config.test_dataset_items,
        mode='test',
        shuffle=config.shuffle,
        max_seq_len=config.max_seq_len,
        pad_id=gpt_neo_config.pad_token_id
    )
    # dataloaders
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=cpu_count(),
        pin_memory=True
    )
    valid_sampler = DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=cpu_count(),
        pin_memory=True
    )
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=cpu_count(),
        pin_memory=True
    )

    # start training
    current_unbinary_ratio = 1.
    while current_unbinary_ratio > config.unbinary_ratio_threshold:
        train_state.epoch += 1
        if local_rank == 0:
            print("=" * 20, f"Training Stage {train_state.stage}", "=" * 20)

        train_state = training(train_dataloader, config, train_state)
        train_state = validation(valid_dataloader, config, train_state)
        train_state = kk_callback(config, train_state)
        current_unbinary_ratio = train_state.unbinary_ratio[-1]


if __name__ == '__main__':
    main(Config())
