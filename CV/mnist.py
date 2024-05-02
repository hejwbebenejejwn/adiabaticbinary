import torch
# from Dense_mnist import DenseNet
from modules.Dense_mnist import DenseNet
# from trainer import Trainer
from modules.trainer import Trainer
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import wandb

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
    root="D:/usr14/project/Binary/wwdata",
    train=True,
    download=False,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root="D:/usr14/project/Binary/wwdata",
    train=False,
    download=False,
    transform=transform,
)

y_train = np.array(train_dataset.targets)

idx = np.argsort(y_train)

vdx = np.array([6000 * i + j for i in range(10) for j in range(5400, 6000)])
tdx = np.array([6000 * i + j for i in range(10) for j in range(5400)])

train_subset = Subset(train_dataset, indices=idx[tdx])
val_subset = Subset(train_dataset, indices=idx[vdx])

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

bw=ba=False
print(bw, ba)
savepath = "D:/usr14/project/Binary/wwdata/mnist1/fullbest.pth"
if bw and not ba:
    mode = "w"
elif ba and not bw:
    mode = "a"
else:
    mode = "b"
mode="n"
print("mnist, mode: " + mode)
model = DenseNet(bw, ba)
# model.load_state_dict(
#     torch.load(savepath)
# )
optz = torch.optim.Adam(model.parameters())
lossfunc = CrossEntropyLoss()
device='cuda' if torch.cuda.is_available() else 'cpu'
trr = Trainer(100, mode, model, optz, lossfunc,0.98,"mnist",device)

print(f"initial accuracy:{trr.evaluate(val_loader=val_loader)}")
trr.train(train_loader,val_loader,500,1e-3,test_loader,savepath,"full precision")

bw=True
ba=False
print(bw, ba)
savepath = "D:/usr14/project/Binary/wwdata/mnist1/binwbest.pth"
if bw and not ba:
    mode = "w"
elif ba and not bw:
    mode = "a"
else:
    mode = "b"
print("mnist, mode: " + mode)
model = DenseNet(bw, ba)
# model.load_state_dict(
#     torch.load(savepath)
# )
optz = torch.optim.Adam(model.parameters())
lossfunc = CrossEntropyLoss()
device='cuda' if torch.cuda.is_available() else 'cpu'
trr = Trainer(100, mode, model, optz, lossfunc,0.98,"mnist",device)

print(f"initial accuracy:{trr.evaluate(val_loader=val_loader)}")
trr.train(train_loader,val_loader,1000,1e-3,test_loader,savepath,"binW")


# if mode == "a":
#     trr.sw_epc = 0
#     trr.lr_base = 1.5
#     trr.lr_power = 0.3
#     model.set_ka(trr.lr_base)
#     trr.train(
#         trainloader=train_loader,
#         valloader=val_loader,
#         max_epochs=8,
#         lnr=1e-3,
#         header="mnist",
#         testloader=test_loader,
#     )
#     model.set_ka(6)
#     trr.train(
#         trainloader=train_loader,
#         valloader=val_loader,
#         max_epochs=3,
#         lnr=1e-3,
#         header="mnist",
#         testloader=test_loader,
#     )
#     model.set_ka(1001)
#     trr.train(
#         trainloader=train_loader,
#         valloader=val_loader,
#         max_epochs=2,
#         lnr=1e-3,
#         header="mnist",
#         testloader=test_loader,
#     )
# if mode == "w":
#     trr.sw_epc = 0
#     trr.lr_base = 3.0
#     model.set_kk(trr.lr_base)
#     trr.train(
#         trainloader=train_loader,
#         valloader=val_loader,
#         max_epochs=8,
#         lnr=1e-3,
#         header="mnist",
#         testloader=test_loader,
#     )
#     for kkz in [10, 20, 50, 100, 300, 500, 999]:
#         model.set_kk(kkz)
#         trr.train(
#             trainloader=train_loader,
#             valloader=val_loader,
#             max_epochs=3,
#             lnr=1e-3,
#             header="mnist",
#             testloader=test_loader,
#         )
# if mode == "b":
#     trr.sw_epc = 0
#     trr.lr_base = 3.0
#     if model.get_kk().item() < 3:
#         model.set_kk(trr.lr_base)
#     if model.get_ka().item() < 3:
#         model.set_ka(trr.lr_base)
#     trr.train(
#         trainloader=train_loader,
#         valloader=val_loader,
#         max_epochs=8,
#         lnr=1e-3,
#         testloader=test_loader,
#         path=savepath,
#     )
#     # for kkz in [10, 20, 50, 100, 300, 500, 999]:
#     #     model.set_ka(kkz)
#     #     model.set_kk(kkz)
#     #     trr.train(
#     #         trainloader=train_loader,
#     #         valloader=val_loader,
#     #         max_epochs=3,
#     #         lnr=1e-3,
#     #         header="mnist",
#     #         testloader=test_loader,
#     #     )
