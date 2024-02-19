import torch
from ww.mnist.Dense_mnist import DenseNet
from ww.trainer import Trainer
from torch.nn import CrossEntropyLoss


val_loader=0
train_loader=0
test_loader=0

bw = True
ba = True
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

print(f'initial accuracy:{trr.evaluate(val_loader=val_loader)}')

if(mode=="a"):
    trr.sw_epc=0
    trr.lr_base=1.5
    trr.lr_power=.3
    model.set_ka(trr.lr_base)
    trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=8,lnr=1e-3,header='mnist',testloader=test_loader)
    model.set_ka(6)
    trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=3,lnr=1e-3,header='mnist',testloader=test_loader)
    model.set_ka(1001)
    trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=2,lnr=1e-3,header='mnist',testloader=test_loader)
if(mode=="w"):
    trr.sw_epc=0
    trr.lr_base=3.0
    model.set_kk(trr.lr_base)
    trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=8,lnr=1e-3,header='mnist',testloader=test_loader)
    for kkz in [10, 20, 50, 100, 300, 500, 999]:
        model.set_kk(kkz)
        trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=3,lnr=1e-3,header='mnist',testloader=test_loader)
if(mode=="b"):
    trr.sw_epc=0
    trr.lr_base=3.0
    model.set_kk(trr.lr_base)
    model.set_ka(trr.lr_base)
    trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=8,lnr=1e-3,header='mnist',testloader=test_loader)
    for kkz in [10, 20, 50, 100, 300, 500, 999]:
        model.set_ka(kkz)
        model.set_kk(kkz)
        trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=3,lnr=1e-3,header='mnist',testloader=test_loader)
