import os
import re
from glob import glob
import time
import wandb
import fire
from typing import Optional, Tuple, List
from multiprocessing import cpu_count
import psutil
import platform
import GPUtil

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from llama import Llama, Tokenizer, Transformer
from binary_llama import BinaryLlama, BinaryTransformer
from datasets import TokenizedDataset

torch.set_float32_matmul_precision('medium')
wandb.login(key='86d6482d3fd7abdbe5d578208634a88905840ce9')

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.device(f"cuda:{local_rank}")


# %% configurations
class Config:
    # dataset config
    tokenized_dataset_path: str = 'datasets/tokenized/SlimPajama-627B'
    dataset_ratio: list = None
    batch_size: int = 128
    shuffle: bool = True
    dtype: torch.dtype = torch.float16

    # model config
    llama_ckpt_path: str = 'models/Meta-Llama-3-8B'
    initial_ckpt_path: str = 'models/Meta-Llama-3-8B(with binary parameters)'
    tokenizer_path: str = 'models/tokenizer.model'
    max_seq_len: int = 512
    max_batch_size: int = batch_size
    model_parallel_size: Optional[int] = 1
    seed: int = 1
    max_gen_len: Optional[int] = None
    block_size: int = max_seq_len - 1
    temperature: float = 0.6
    top_p: float = 0.9

    # training config
    unbinary_ratio_threshold: float = 0.005  # training termination threshold
    kk_threshold: float = 100.  # kk threshold to divide training stage 1 and stage 2
    accum_batches: int = 16
    betas: Tuple[float, float] = (0.9, 0.95)
    base_lr: float = 1e-4
    base_step_size: int = 100
    base_gamma: float = 0.9
    save_path: str = 'models/BinaryLlama'
    file_prefix: str = 'binary-llama'

    # binary KD training config
    kd_temperature: float = 10.
    kd_alpha: float = 0.1
    aa_lr: float = 0.4
    kk_lr: float = 1.25
    ratio: float = 0.1
    patience = 15


class TrainingState:
    # model state
    binary_llama: BinaryLlama
    llama: Llama
    llama_model: Transformer
    model: BinaryTransformer
    loss_fn1: F.cross_entropy
    loss_fn2: F.mse_loss
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


# %% setup
def build_model(config: Config = Config()) -> Tuple[BinaryLlama, Llama]:
    # build binary model
    binary_llama = BinaryLlama.build(ckpt_dir=config.initial_ckpt_path, tokenizer_path=config.tokenizer_path,
                                     max_seq_len=config.max_seq_len, max_batch_size=config.max_batch_size,
                                     model_parallel_size=config.model_parallel_size, seed=config.seed)
    # build llama model
    llama = Llama.build(ckpt_dir=config.llama_ckpt_path, tokenizer_path=config.tokenizer_path,
                        max_seq_len=config.max_seq_len, max_batch_size=config.max_batch_size,
                        model_parallel_size=config.model_parallel_size, seed=config.seed)
    return binary_llama, llama


def set_up(config: Config = Config()) -> None:
    """Build model and Initialize optimizer & learning rate scheduler"""
    binary_llama, llama = build_model(config)
    binary_model = binary_llama.model
    llama_model = llama.model

    loss_fn1 = F.cross_entropy
    loss_fn2 = F.mse_loss
    optimizer = torch.optim.Adam(binary_model.parameters(), betas=config.betas, lr=config.base_lr)
    # change base_step_size & base_gamma as current lr changes
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.base_step_size, gamma=config.base_gamma)

    TrainingState.binary_llama = binary_llama
    TrainingState.llama = llama
    TrainingState.model = binary_model
    TrainingState.llama_model = llama_model
    TrainingState.loss_fn1 = loss_fn1
    TrainingState.loss_fn2 = loss_fn2
    TrainingState.optimizer = optimizer
    TrainingState.lr_scheduler = lr_scheduler


# %% training
def training_step(batch, config: Config = Config()) -> torch.Tensor:
    """Training Step"""
    binary_llama = TrainingState.binary_llama
    llama = TrainingState.llama
    loss_fn1 = TrainingState.loss_fn1
    loss_fn2 = TrainingState.loss_fn2

    input_tokens = batch[:, 0:config.block_size].contiguous()
    target_tokens = batch[:, 1:config.block_size + 1].contiguous().reshape(-1).view(-1).long()

    binary_logits, _ = binary_llama.gen(prompt_tokens=input_tokens, temperature=config.temperature, logprobs=True)
    binary_logits = binary_logits.to(device=device, dtype=torch.float32).view(-1, binary_logits.size(-1))

    _, llama_kd_probs = llama.gen(prompt_tokens=input_tokens, temperature=config.kd_temperature, logprobs=True)
    llama_kd_probs = llama_kd_probs.to(device=device, dtype=torch.float32).view(-1, llama_kd_probs.size(-1))

    _, binary_kd_probs = binary_llama.gen(prompt_tokens=input_tokens, temperature=config.kd_temperature, logprobs=True)
    binary_kd_probs = binary_kd_probs.to(device=device, dtype=torch.float32).view(-1, binary_kd_probs.size(-1))

    # logit KD loss
    loss_ce = loss_fn1(input=binary_logits, target=target_tokens, ignore_index=-1)
    loss_mse = loss_fn2(input=binary_kd_probs, target=llama_kd_probs)
    loss = loss_ce + loss_mse * config.kd_alpha

    # feature KD loss
    return loss


def training(train_dataloader: DataLoader, config: Config = Config()) -> None:
    """Training Epoch"""
    optimizer = TrainingState.optimizer
    lr_scheduler = TrainingState.lr_scheduler
    model = TrainingState.model
    model.train()
    scaler = GradScaler()

    # training loops
    if local_rank == 0:
        kk = TrainingState.kk
        aa = TrainingState.aa
        print(f"[GPU{local_rank}] Epoch {TrainingState.epoch}, kk = {kk:.2f}, aa = {aa:.2f} Training "
              + "=" * 10, flush=True)
    start = 0

    train_epoch_loss = []
    for i, data in enumerate(train_dataloader):
        data = data.to(device, non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            train_loss = training_step(data, config) / config.accum_batches
        scaler.scale(train_loss).backward()

        if (i + 1) % config.accum_batches == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            TrainingState.accum_step += 1
            n_accum = TrainingState.accum_step

            if local_rank == 0:
                elapsed = time.time() - start
                print(f"Epoch Step: {i + 1:6d} | Accumulation Step: {n_accum:3d} | Loss: {train_loss:6.2f} " +
                      f"| Tokens/Sec: {config.max_seq_len/elapsed:7.1f} | Learning Rate: {lr_scheduler.get_lr():6.1e}")
                start = time.time()

        train_epoch_loss.append(train_loss)

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
    target_tokens = batch[:, 1:config.block_size + 1].contiguous().view(-1).long()
    binary_llama = TrainingState.binary_llama
    loss_fn1 = TrainingState.loss_fn1

    valid_logits, _ = binary_llama.gen(prompt_tokens=input_tokens, temperature=config.temperature, logprobs=True)
    valid_logits = valid_logits.to(device=device, dtype=torch.float32).view(-1, valid_logits.size(-1))

    valid_loss = loss_fn1(input=valid_logits, target=target_tokens, ignore_index=-1)
    return valid_loss


def validation(valid_dataloader: DataLoader, config: Config = Config()) -> None:
    """Validation Epoch"""
    model = TrainingState.model
    model.eval()

    kk = TrainingState.kk
    aa = TrainingState.aa
    if local_rank == 0:
        print(f"[GPU{local_rank}] Epoch {TrainingState.epoch}, kk = {kk:.2f}, aa = {aa:.2f} Validation "
              + "=" * 10, flush=True)
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
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
    for name, param in model.named_parameters():
        if name.endswith('.weight') and name != 'tok_embeddings.weight':
            weight = param.flatten()
            k = model.state_dict()[name[:-7] + '.kk']
            a = model.state_dict()[name[:-7] + '.aa']
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
    file_path = f'{config.save_path}/{config.file_prefix}-{current_epoch}.pt'
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
        file_to_del = [f'{config.save_path}/{config.file_prefix}-{i}.pt'
                       for i in range(TrainingState.best_epoch + 1, current_epoch + 1)]
        for file in file_to_del:
            os.remove(file)
        # load best state
        best_file = f'{config.save_path}/{config.file_prefix}-{TrainingState.best_epoch}.pt'
        checkpoint = torch.load(best_file)['train_state']
        # %% new state
        for attr in TrainingState.__dict__.keys():
            setattr(TrainingState, attr, checkpoint[attr])
        kk = TrainingState.model.state_dict()['layers.0.attention.wq.kk']

        # push kk and aa
        if kk < 1:
            kk += config.aa_lr
            aa = 1 / kk
        else:
            kk *= config.kk_lr
            aa = torch.tensor([1.], dtype=kk.dtype, device=device)

        # stage 1
        if kk < config.kk_threshold:
            TrainingState.model.set_kk_stage1(torch.tensor([kk, aa], dtype=kk.dtype, device=device))
        # stage 2
        else:
            TrainingState.model.set_kk_stage2(torch.tensor(config.ratio, dtype=kk.dtype, device=device))

        # record changes
        kk = TrainingState.model.state_dict()['layers.0.attention.wq.kk']
        aa = TrainingState.model.state_dict()['layers.0.attention.wq.aa']
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
def main(config: Config = Config()) -> None:
    # initialize
    wandb.init(project='binary-llama')
    wandb.config.update(config)
    os.makedirs(config.save_path, exist_ok=True)

    # show platform information
    if local_rank == 0:
        print("=" * 20, "Platform Information", "=" * 20)
        cpu_model = platform.processor()
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024 ** 3)
        memory_usage_percent = memory_info.percent
        cpu_threads = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU: {cpu_model}")
        print(f"Memory: {total_memory_gb:.2f} GB")
        print(f"Memory usage: {memory_usage_percent:.2f}%")
        print(f"CPU threads: {cpu_threads}")
        print(f"CPU usage: {cpu_usage}%\n")
        gpus = GPUtil.getGPUs()
        i = 0
        for gpu in gpus:
            print(f"GPU{i}: {gpu.name}")
            print(f"Memory Total: {gpu.memoryTotal / 1024:.2f} GB")
            i += 1
        GPUtil.showUtilization()

    # prepare model
    if local_rank == 0:
        print("=" * 20, "Preparing Model", "=" * 20)
    set_up(config)

    # check if resume
    ckpt_files = glob(f'{config.save_path}/*.pt', recursive=True)
    latest_checkpoint = max(ckpt_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)), default=None)
    if latest_checkpoint:
        if local_rank == 0:
            print("=" * 20, f"Resume from {latest_checkpoint}", "=" * 20)
        checkpoint = torch.load(latest_checkpoint)['train_state']
        for attr in TrainingState.__dict__.keys():
            setattr(TrainingState, attr, checkpoint[attr])

    # prepare data
    # datasets
    tokenizer = Tokenizer(config.tokenizer_path)
    train_dataset = TokenizedDataset(tokenized_dataset_path=config.tokenized_dataset_path,
                                     dataset_ratio=config.dataset_ratio, mode='training', shuffle=config.shuffle,
                                     max_seq_len=config.max_seq_len, pad_id=tokenizer.pad_id)
    valid_dataset = TokenizedDataset(tokenized_dataset_path=config.tokenized_dataset_path,
                                     dataset_ratio=config.dataset_ratio, mode='validation', shuffle=config.shuffle,
                                     max_seq_len=config.max_seq_len, pad_id=tokenizer.pad_id)
    test_dataset = TokenizedDataset(tokenized_dataset_path=config.tokenized_dataset_path,
                                    dataset_ratio=config.dataset_ratio, mode='test', shuffle=config.shuffle,
                                    max_seq_len=config.max_seq_len, pad_id=tokenizer.pad_id)
    # dataloaders
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
    model = TrainingState.model
    model.to(device)
    llama_model = TrainingState.llama_model
    llama_model.to(device)
    llama_model.eval()  # set initial llama model evaluation only

    # make ddp model
    TrainingState.llama_model = DDP(llama_model, device_ids=[local_rank], output_device=local_rank)
    TrainingState.model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
    fire.Fire(main)
