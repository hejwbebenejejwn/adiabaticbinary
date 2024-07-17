import sys
import os

CHECKPOINTS_DIR = "/data/home/wwang/projs/binary/checkpoint/imagenet"
MODULES_DIR = "/data/home/wwang/projs/binary/adiabaticbinary/CV"

sys.path.insert(0, MODULES_DIR)
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
MAGENTA = "\033[95m"


from modules import resnet1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from readdata import read_dataset
import wandb

# wandb.login(key="86d6482d3fd7abdbe5d578208634a88905840ce9")
device = torch.device("cuda")


def fit(model: resnet1.ResNet, optim, lossfunc, trainloader: DataLoader):
    model.train()
    totalloss = 0
    for data, target in trainloader:
        data, target = data.cuda(), target.cuda()
        optim.zero_grad()
        output = model(data)
        loss = lossfunc(output, target)
        loss.backward()
        optim.step()
        with torch.no_grad():
            totalloss += loss.item() * data.size(0)
    return totalloss / len(trainloader.sampler)


def evaluate(
    model: resnet1.ResNet, val_loader: DataLoader, lossfunc: nn.CrossEntropyLoss
):
    model.eval()
    loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss += lossfunc(outputs, labels).item() * inputs.size(0)
            _, predicted_top1 = torch.max(outputs, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()
            _, predicted_top5 = outputs.topk(5, 1, True, True)
            correct_top5 += (predicted_top5 == labels.view(-1, 1)).sum().item()
            total += labels.size(0)
        loss /= len(val_loader.sampler)
        acc_top1 = correct_top1 / total
        acc_top5 = correct_top5 / total
    return loss, acc_top1, acc_top5


model = resnet1.ResNet(True, 1000, False)
dicfull = torch.load(os.path.join(CHECKPOINTS_DIR, "lr000039kk15acc54.pth"))
dicfull = {k.replace('module.',''):v for k,v in dicfull.items()}
dicbin = model.state_dict()
dicbin.update(dicfull)
model.load_state_dict(dicbin)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
else:
    model.cuda()
    print("Using single GPU")

lossfunc = nn.CrossEntropyLoss().to(device)
train_loader, val_loader, test_loader = read_dataset(128, subset=False, num_workers=16)

# wb = wandb.init(project="imagenet1k binary", name="first run", reinit=True)
wb = False

target_acc = 0.67
mepoch = 99999999
mstep_kk = 25
mstep_lr = 3
lr = .1/256
kk_mul = 1.5

_, val_acc1, val_acc5 = evaluate(model, val_loader, lossfunc)
print(f"last fullbest, top1:{val_acc1}, top5:{val_acc5}")
while val_acc1 > target_acc:
    model.module.set_kk(model.module.get_kk() * kk_mul)
    _, val_acc1, val_acc5 = evaluate(model, val_loader, lossfunc)
    print(f"push kk to {model.module.get_kk()}, top1:{val_acc1}, top5:{val_acc5}")
kk_now, ka_now = model.module.toBin()
bin_loss, binbest1, binbest5 = evaluate(model, val_loader, lossfunc)
print(f"last binbest, top1:{binbest1}, top5{binbest5}, ka:{ka_now}, kk:{kk_now}")
model.module.quitBin()
if binbest1 >= target_acc:
    print(RED + "target acc already reached" + RESET)
    sys.exit(0)

step_kk = 0
step_lr = 0
val_loss_best = np.inf
val_top1_best = 0
val_top5_best = 0

for epoch in range(mepoch):
    if step_lr / mstep_lr  == 1:
        step_lr = 0
        lr *= 0.5
        print(GREEN + f"lr reduced to {lr}" + RESET)
    optim = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=5e-4)

    train_loss = fit(model, optim, lossfunc, train_loader)
    val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, lossfunc)

    if val_loss < val_loss_best:
        step_kk -= step_lr + 1
        step_lr = 0
        val_loss_best = val_loss
    else:
        step_lr += 1

    model.module.toBin()
    bin_loss, bintop1, bintop5 = evaluate(model, val_loader, lossfunc)
    if binbest1 < bintop1:
        binbest1 = bintop1
        binbest5 = bintop5
        _, testtop1, testtop5 = evaluate(model, test_loader, lossfunc)
        print(BLUE + f"test perf, top1:{testtop1}, top5:{testtop5}" + RESET)
        model.module.quitBin()
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINTS_DIR, "binbest.pth"),
        )
    else:
        model.module.quitBin()

    if binbest1 > target_acc:
        print(RED + "target acc reached" + RESET)
        break

    if val_top1_best < val_acc1:
        val_top1_best = val_acc1
        val_top5_best = val_acc5
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "temp.pth"))
    if step_kk >= mstep_kk:
        val_acc1 = target_acc = val_top1_best
        val_acc5 = val_top5_best
        val_top1_best = 0
        step_kk = 0
        print(RED + f"reduce target acc to {target_acc}" + RESET)

    print(
        "epoch",
        epoch + 1,
        "train_loss",
        train_loss,
        "val_loss",
        val_loss,
        "val_top1",
        val_acc1,
        "val_top5",
        val_acc5,
        "bin_top1",
        bintop1,
        "bin_top5",
        bintop5,
        "steplr",
        step_lr,
        "step_kk",
        step_kk,
        "lr",
        round(lr, 5),
        "kk",
        round(model.module.get_kk(), 3),
        "tg",
        round(target_acc, 3),
    )
    if wb:
        wb.log(
            {
                "epoch": epoch + 1,
                "train loss": train_loss,
                "val top1": val_acc1,
                "val top5": val_acc5,
                "bin top1": bintop1,
                "bin top5": bintop5,
                "lr": round(lr, 5),
                "kk": round(model.get_kk(), 3),
                "tg": round(target_acc, 3),
            }
        )
    if val_acc1 >= target_acc - 1e-4:
        model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "temp.pth")))
        step_kk = step_lr = val_top1_best = 0
        val_loss_best = np.inf
        lr = 0.1
        while val_acc1 > target_acc - 1e-4:
            model.module.set_kk(model.module.get_kk() * kk_mul)
            _, val_acc1, val_acc5 = evaluate(model,val_loader,lossfunc)
        print(YELLOW + f"push kk to {model.module.get_kk()}, top1:{val_acc1}, top5:{val_acc5}" + RESET)
    else:
        step_kk += 1
if wb:
    wb.finish()
