import numpy as np
import torch
import torch.nn as nn
from modules.base import Base
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        same_wts_ep,
        mode,
        model: Base,
        optim: torch.optim.Adam,
        lossfunc: nn.CrossEntropyLoss,
    ):
        self.model = model
        self.optim = optim
        self.lossfunc = lossfunc
        self.target_acc = 1.0
        self.binbest = 0.0
        self.same_wts_ep = same_wts_ep
        self.sw_epc = 0
        self.mode = mode
        self.pmode = "a" if mode == "a" else "w"
        self.prto = 1.5
        self.maxpush = 3
        self.lr_power = 0.3
        self.lr_base = 1.0
        self.val_bs = 125

    def refresh(self):
        self.target_acc = 1.0
        self.binbest = 0.0
        self.sw_epc = 0

    def fit(self, trainloader: DataLoader):
        self.model.train()
        for _, (data, target) in enumerate(trainloader):
            self.optim.zero_grad()
            output = self.model(data)
            loss = self.lossfunc(output, target)
            loss.backward()
            self.optim.step()

    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
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
        header,
        testloader,
    ):
        val_best = 0
        for epoch_i in range(max_epochs):
            lr = (
                lnr / (self.model.get_kk() / self.lr_base) ** self.lr_power
                if self.mode == "w"
                else lnr / (self.model.get_ka() / self.lr_base) ** self.lr_power
            )
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            self.sw_epc += 1
            print(
                "epoch",
                epoch_i + 1,
                "sw",
                self.sw_epc,
                "mxep",
                max_epochs,
                "tg",
                self.target_acc,
                "bb",
                self.binbest,
            )
            vala = self.fit(trainloader)
            val_best = max(vala, val_best)

            if self.mode == "b" or self.mode == "w":
                kk_now = torch.clone(self.model.get_kk()).item()
                self.model.set_kk(1e5)
            if self.mode == "b" or self.mode == "a":
                ka_now = torch.clone(self.model.get_ka()).item()
                self.model.set_ka(1e5)

            vbin = self.evaluate(valloader)

            if self.binbest < vbin:
                self.binbest = vbin
                torch.save(self.model.state_dict(), 'C:/Projects/Binary/adiabaticbinary/ww/'+header+'/binbest.pth')
                print("\033[94mtest perf: ",end="")
                self.evaluate(testloader)
                print("\033[0m",end="")

            if self.mode == "b" or self.mode == "w":
                self.model.set_kk(kk_now)
            if self.mode == "b" or self.mode == "a":
                self.model.set_ka(ka_now)

            if self.binbest >= self.target_acc:
                break

            if self.sw_epc >= self.same_wts_ep:
                vala = self.target_acc = val_best
                self.sw_epc = 0
                print("reduce acc to", val_best)

            if vala >= self.target_acc:
                val_best = 0
                if self.pmode == "w":
                    for _ in range(self.maxpush):
                        if vala < self.target_acc or self.model.get_kk().item() > 1e3:
                            break
                        self.model.set_kk(self.model.get_kk() * self.prto)
                        vala = self.evaluate(valloader)
                        print("push kk to", self.model.get_kk().item(), "acc=", vala)
                    if self.mode == "b":
                        self.pmode = "a"
                if self.pmode == "a":
                    for _ in range(self.maxpush):
                        if vala < self.target_acc or self.model.get_ka().item() > 1e3:
                            break
                        self.model.set_ka(self.model.get_ka() * self.prto)
                        vala = self.evaluate(valloader)
                        print("push ka to", self.model.get_ka().item(), "acc=", vala)
                    if self.mode == "b":
                        self.pmode = "w"
                self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
                self.sw_epc = 0

            if self.mode == "b" or self.mode == "w":
                if self.model.get_kk().item() > 1e3:
                    break
            if self.mode == "b" or self.mode == "a":
                if self.model.get_ka().item() > 1e3:
                    break
