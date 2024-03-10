import numpy as np
import torch
import torch.nn as nn
import wandb
import modules.base as base

# import base
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        same_wts_ep,
        mode,
        model: base.Base,
        optim: torch.optim.Adam,
        lossfunc: nn.CrossEntropyLoss,
        targetacc=1,
        project=None,
        device="cpu",
    ):
        self.model = model
        self.optim = optim
        self.lossfunc = lossfunc
        self.target_acc = targetacc
        self.same_wts_ep = same_wts_ep
        self.sw_epc = 0
        self.mode = mode
        self.pmode = "a" if mode == "a" else "w"
        if self.mode == "n":
            self.pmode = self.mode
        self.prto = 1.5
        self.maxpush = 3
        self.lr_power = 0.3
        self.lr_base = 1.0
        self.device = torch.device(device)
        self.model.to(device)
        self.wandb = None
        self.project = project

    def refresh(self):
        self.target_acc = 1.0
        self.binbest = 0.0
        self.sw_epc = 0

    def fit(self, trainloader: DataLoader):
        self.model.train()
        totalloss = 0
        for data, target in trainloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            output = self.model(data)
            loss = self.lossfunc(output, target)
            with torch.no_grad():
                totalloss += loss.item()
            loss.backward()
            self.optim.step()
        return totalloss

    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def train(
        self,
        trainloader,
        valloader,
        max_epochs,
        lnr,
        testloader,
        path,
        notes=None,
        tags=None,
    ):

        kk_now, ka_now = self.model.toBin()
        self.binbest = self.evaluate(valloader)
        print(f"last binbest:{self.binbest}, ka:{ka_now}, kk:{kk_now}")
        self.model.quitBin()

        val_best = 0
        if self.binbest >= self.target_acc:
            print("\033[91mtarget acc already reached")
            print("\033[0m")
            return None

        if self.project:
            self.wandb = wandb.init(
                project=self.project, notes=notes, tags=tags, reinit=True
            )
            config = self.wandb.config
            config.lnr = lnr
            config.bw = int(self.binW)
            config.ba = int(self.binA)
            config.maxepoch = max_epochs

        for epoch_i in range(max_epochs):
            if self.mode == "n":
                lr = lnr
            else:
                lr = (
                    lnr / (self.model.get_kk() / self.lr_base) ** self.lr_power
                    if self.mode == "w"
                    else lnr / (self.model.get_ka() / self.lr_base) ** self.lr_power
                ).item()
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            self.sw_epc += 1
            print(
                "epoch",
                epoch_i + 1,
                "sw",
                self.sw_epc,
                "maxep",
                max_epochs,
                "tg",
                round(self.target_acc, 3),
                "bbest",
                round(self.binbest, 3),
                "vbest",
                round(val_best, 3),
                "kk",
                (" " if not self.model.binW else round(self.model.get_kk().item(), 3)),
                "ka",
                (" " if not self.model.binA else round(self.model.get_ka().item(), 3)),
            )
            loss = self.fit(trainloader)
            vala = self.evaluate(valloader)
            val_best = max(vala, val_best)

            self.model.toBin()
            vbin = self.evaluate(valloader)

            if self.binbest < vbin:
                self.binbest = vbin
                print("\033[94mtest perf: ", end="")
                print(self.evaluate(testloader))
                print("\033[0m")
                self.model.quitBin()
                torch.save(
                    self.model.state_dict(),
                    path + "binbest.pth",
                )
            else:
                self.model.quitBin()

            if self.wandb:
                self.wandb.log(
                    {
                        "epoch": epoch_i + 1,
                        "kk": 0 if not self.model.binW else self.model.get_kk(),
                        "ka": 0 if not self.model.binA else self.model.get_ka(),
                        "loss": loss,
                        "val acc": vala,
                        "val binacc": vbin,
                        "target acc": self.target_acc,
                    }
                )

            if self.binbest >= self.target_acc:
                print("\033[91mtarget acc reached")
                print("\033[0m")
                break

            if self.sw_epc >= self.same_wts_ep:
                vala = self.target_acc = val_best
                self.sw_epc = 0
                print("reduce acc to", val_best)

            if vala >= self.target_acc:
                if self.pmode == "w":
                    for _ in range(self.maxpush):
                        if vala < self.target_acc or self.model.get_kk().item() > 1e3:
                            break
                        self.model.set_kk(self.model.get_kk().item() * self.prto)
                        vala = self.evaluate(valloader)
                        print("push kk to", self.model.get_kk().item(), "acc=", vala)
                    if self.mode == "b":
                        self.pmode = "a"
                if self.pmode == "a":
                    for _ in range(self.maxpush):
                        if vala < self.target_acc or self.model.get_ka().item() > 1e3:
                            break
                        self.model.set_ka(self.model.get_ka().item() * self.prto)
                        vala = self.evaluate(valloader)
                        print("push ka to", self.model.get_ka().item(), "acc=", vala)
                    if self.mode == "b":
                        self.pmode = "w"
                self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
                self.sw_epc = 0

            if self.mode == "w":
                if self.model.get_kk().item() > 1e3:
                    print("\033[91mweights binaried")
                    print("\033[0m")
                    torch.save(
                        self.model.state_dict(),
                        path + "binw.pth",
                    )
                    break
            if self.mode == "a":
                if self.model.get_ka().item() > 1e3:
                    print("\033[91mactivations binaried")
                    print("\033[0m")
                    torch.save(
                        self.model.state_dict(),
                        path + "bina.pth",
                    )
                    break
            if self.mode == "b":
                if self.model.get_kk().item() > 1e3:
                    print("\033[91mweights binaried")
                    print("\033[0m")
                    kbin = True
                else:
                    kbin = False
                if self.model.get_ka().item() > 1e3:
                    print("\033[91mactivations binaried")
                    print("\033[0m")
                    abin = True
                else:
                    abin = False
                if kbin and abin:
                    torch.save(
                        self.model.state_dict(),
                        path + "binbest.pth",
                    )
                    break


if __name__ == "__main__":
    import torch

    from Conv_DogCat import CatDog

    # from modules.Conv_DogCat import CatDog

    from trainer import Trainer

    # from modules.trainer import Trainer
    from torch.nn import CrossEntropyLoss
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    def catdog_datagen(path):
        """dog:0 cat:1"""
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = datasets.ImageFolder(root=path, transform=transform)

        cat_class_idx = dataset.class_to_idx["cats"]
        dog_class_idx = dataset.class_to_idx["dogs"]

        if cat_class_idx < dog_class_idx:
            dataset.targets = [
                1 if label == cat_class_idx else 0 for label in dataset.targets
            ]
            dataset.class_to_idx["dogs"], dataset.class_to_idx["cats"] = (
                dataset.class_to_idx["cats"],
                dataset.class_to_idx["dogs"],
            )

        return dataset

    traval = catdog_datagen("C:/Projects/Binary/wwdata/dogs_vs_cats/train")
    train_size = int(0.75 * len(traval))
    val_size = len(traval) - train_size
    train_dataset, val_dataset = random_split(traval, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    testset = catdog_datagen("C:/Projects/Binary/wwdata/dogs_vs_cats/test")
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    bw = ba = True
    print(bw, ba)
    savepath = "C:/Projects/Binary/wwdata/dogcat/"
    if bw and not ba:
        mode = "w"
    elif ba and not bw:
        mode = "a"
    else:
        mode = "b"
    print("dogcat, mode: " + mode)
    model = CatDog(bw, ba)
    # model.load_state_dict(torch.load(savepath))
    optz = torch.optim.Adam(model.parameters())
    lossfunc = CrossEntropyLoss()
    trr = Trainer(1000, mode, model, optz, lossfunc, 0.9)

    if mode == "w":
        bw = 1
        ba = 0
        initLR = 0.1
        optz = torch.optim.Aedam(model.parameters(), lr=initLR)

    elif mode == "a":
        bw = 0
        ba = 1
        initLR = 0.01
        optz = torch.optim.Adam(model.parameters(), lr=initLR)
        trr.lr_power = 0.3
    elif mode == "b":
        bw = ba = 1
        initLR = 0.01
        optz = torch.optim.Adam(model.parameters(), lr=initLR)
        trr.lr_power = 0.3

    trr.train(
        trainloader=train_loader,
        valloader=val_loader,
        max_epochs=3,
        lnr=1e-3,
        testloader=test_loader,
        path=savepath,
    )
