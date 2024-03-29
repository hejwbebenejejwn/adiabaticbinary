import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
from torch.nn import CrossEntropyLoss
import numpy as np
from modules.ResNet import ResNet20
from modules.trainer import Trainer

data_dir='D:/usr14/project/Binary/imagenet1/train'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

dataset=torchvision.datasets.ImageFolder(root=data_dir,transform=transform)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size=len(dataset)-train_size-val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

bw=ba=False
print(bw, ba)
savepath = "D:/usr14/project/Binary/wwdata/imagenet/fullbest.pth"
if bw and not ba:
    mode = "w"
elif ba and not bw:
    mode = "a"
else:
    mode = "b"
mode="n"
print("mnist, mode: " + mode)
model = ResNet20(bw, ba)
# model.load_state_dict(
#     torch.load(savepath)
# )
optz = torch.optim.Adam(model.parameters(),lr=0.01)
lossfunc = CrossEntropyLoss()
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"device {device}")
trr = Trainer(10000, mode, model, optz, lossfunc,0.98,"imagenet",device)

print(f"initial accuracy:{trr.evaluate(val_loader=val_loader)}")
trr.train(train_loader,val_loader,200,1e-3,test_loader,savepath,"full precision")

bw=True
ba=False
print(bw, ba)
savepath = "D:/usr14/project/Binary/wwdata/imagenet/binwbest.pth"
if bw and not ba:
    mode = "w"
elif ba and not bw:
    mode = "a"
else:
    mode = "b"
print("mnist, mode: " + mode)
model = ResNet20(bw, ba)
# model.load_state_dict(
#     torch.load(savepath)
# )
optz = torch.optim.Adam(model.parameters(),lr=0.01)
lossfunc = CrossEntropyLoss()
device='cuda' if torch.cuda.is_available() else 'cpu'
trr = Trainer(50, mode, model, optz, lossfunc,1,"imagenet",device)

print(f"initial accuracy:{trr.evaluate(val_loader=val_loader)}")
trr.train(train_loader,val_loader,500,1e-3,test_loader,savepath,"binW")


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