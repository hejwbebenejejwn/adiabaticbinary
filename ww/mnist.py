import torch
from modules.Dense_mnist import DenseNet
from modules.trainer import Trainer
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='C:/Projects/Binary/wwdata', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='C:/Projects/Binary/wwdata', train=False, download=False, transform=transform)

y_train = np.array(train_dataset.targets)

idx = np.argsort(y_train)

vdx = np.array([6000*i+j for i in range(10) for j in range(5400, 6000)])
tdx = np.array([6000*i+j for i in range(10) for j in range(5400)])

train_subset = Subset(train_dataset, indices=idx[tdx])
val_subset = Subset(train_dataset, indices=idx[vdx])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


parser = ArgumentParser()
parser.add_argument("-w", "--binw", action="store_true")
parser.add_argument("-a", "--bina", action="store_true")
args = parser.parse_args()
bw, ba = args.binw, args.bina
print(bw,ba)

if bw and not ba:
    mode = "w"
elif ba and not bw:
    mode = "a"
else:
    mode = "b"
print("mnist, mode: " + mode)
model = DenseNet(bw, ba)
optz = torch.optim.Adam(model.parameters())
lossfunc = CrossEntropyLoss()
trr = Trainer(1000, mode, model, optz, lossfunc)
trr.val_bs = 32
trr.lr_power = 0.2

print(f"initial accuracy:{trr.evaluate(val_loader=val_loader)}")

if mode == "a":
    trr.sw_epc = 0
    trr.lr_base = 1.5
    trr.lr_power = 0.3
    model.set_ka(trr.lr_base)
    trr.train(
        trainloader=train_loader,
        valloader=val_loader,
        max_epochs=8,
        lnr=1e-3,
        header="mnist",
        testloader=test_loader,
    )
    model.set_ka(6)
    trr.train(
        trainloader=train_loader,
        valloader=val_loader,
        max_epochs=3,
        lnr=1e-3,
        header="mnist",
        testloader=test_loader,
    )
    model.set_ka(1001)
    trr.train(
        trainloader=train_loader,
        valloader=val_loader,
        max_epochs=2,
        lnr=1e-3,
        header="mnist",
        testloader=test_loader,
    )
if mode == "w":
    trr.sw_epc = 0
    trr.lr_base = 3.0
    model.set_kk(trr.lr_base)
    trr.train(
        trainloader=train_loader,
        valloader=val_loader,
        max_epochs=8,
        lnr=1e-3,
        header="mnist",
        testloader=test_loader,
    )
    for kkz in [10, 20, 50, 100, 300, 500, 999]:
        model.set_kk(kkz)
        trr.train(
            trainloader=train_loader,
            valloader=val_loader,
            max_epochs=3,
            lnr=1e-3,
            header="mnist",
            testloader=test_loader,
        )
if mode == "b":
    trr.sw_epc = 0
    trr.lr_base = 3.0
    model.set_kk(trr.lr_base)
    model.set_ka(trr.lr_base)
    trr.train(
        trainloader=train_loader,
        valloader=val_loader,
        max_epochs=8,
        lnr=1e-3,
        header="mnist",
        testloader=test_loader,
    )
    for kkz in [10, 20, 50, 100, 300, 500, 999]:
        model.set_ka(kkz)
        model.set_kk(kkz)
        trr.train(
            trainloader=train_loader,
            valloader=val_loader,
            max_epochs=3,
            lnr=1e-3,
            header="mnist",
            testloader=test_loader,
        )
