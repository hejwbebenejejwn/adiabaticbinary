import sys
import os

CHECKPOINTS_DIR="/data/home/wwang/projs/binary/checkpoint/imagenet"
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

device = torch.device("cuda")


def fit(model: resnet1.ResNet, optim, lossfunc, trainloader: DataLoader):
    model.train()
    totalloss = 0
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = model(data).to(device)
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
            inputs, labels = inputs.to(device), labels.to(device)
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


model = resnet1.ResNet(False, 1000, False).to(device)
# model.load_state_dict(torch.load("pre1full.pth"))
lossfunc = nn.CrossEntropyLoss().to(device)
train_loader, val_loader, test_loader = read_dataset(32, subset=False, num_workers=2)

lr = 0.1
counter = 0
min_val_loss1 = np.inf
min_val_loss5 = np.inf
for epoch in range(500):
    if counter / 10 == 1:
        counter = 0
        lr *= 0.5
        print(GREEN + f"lr reduced to {lr}" + RESET)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss = fit(model, optim, lossfunc, train_loader)
    val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, lossfunc)
    print(
        f"epoch: {epoch+1}, loss: {loss}, val_loss: {val_loss}, top1_acc:{val_acc1}, top5_acc:{val_acc5}"
    )
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        print(MAGENTA + f"val loss reduced to {min_val_loss},lr {lr}" + RESET)
        counter = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR,"pre1full.pth"))
        with open("lr", "w") as file:
            file.write(f"lr={lr}")
    else:
        counter += 1
