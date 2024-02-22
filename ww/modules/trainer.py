import numpy as np
import torch
import torch.nn as nn

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
        self.prto = 1.5
        self.maxpush = 3
        self.lr_power = 0.3
        self.lr_base = 1.0
        self.val_bs = 125
        self.device = torch.device(device)
        self.model.to(device)

    def refresh(self):
        self.target_acc = 1.0
        self.binbest = 0.0
        self.sw_epc = 0

    def fit(self, trainloader: DataLoader):
        self.model.train()
        for data, target in trainloader:
            data, target = data.to(self.device), target.to(self.device)
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def train(self, trainloader, valloader, max_epochs, lnr, testloader, path):
        kk_now, ka_now = self.model.toBin()
        self.binbest = self.evaluate(valloader)
        print(f"last binbest:{self.binbest}, ka:{ka_now}, kk:{kk_now}")
        self.model.quitBin()

        val_best = 0
        if self.binbest >= self.target_acc:
            print("already reached")
            return None

        for epoch_i in range(max_epochs):
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
                " " if self.mode == "a" else round(self.model.get_kk().item(), 3),
                "ka",
                " " if self.mode == "w" else round(self.model.get_ka().item(), 3),
            )
            self.fit(trainloader)
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
                    path,
                )
            else:
                self.model.quitBin()

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

            if self.mode == "b" or self.mode == "w":
                if self.model.get_kk().item() > 1e3:
                    break
            if self.mode == "b" or self.mode == "a":
                if self.model.get_ka().item() > 1e3:
                    break


if __name__ == "__main__":
    from Dense_mnist import DenseNet
    from trainer import Trainer
    from torch.nn import CrossEntropyLoss
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="C:/Projects/Binary/wwdata",
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="C:/Projects/Binary/wwdata",
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

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    bw = ba = True
    print(bw, ba)
    savepath = "C:/Projects/Binary/wwdata/mnist1/binbest1.pth"
    if bw and not ba:
        mode = "w"
    elif ba and not bw:
        mode = "a"
    else:
        mode = "b"
    print("mnist, mode: " + mode)
    model = DenseNet(bw, ba)
    model.load_state_dict(
        torch.load(savepath)
    )
    optz = torch.optim.Adam(model.parameters())
    lossfunc = CrossEntropyLoss()
    trr = Trainer(1000, mode, model, optz, lossfunc,0.95)
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
        if model.get_kk().item() < 3:
            model.set_kk(trr.lr_base)
        if model.get_ka().item() < 3:
            model.set_ka(trr.lr_base)
        trr.train(
            trainloader=train_loader,
            valloader=val_loader,
            max_epochs=8,
            lnr=1e-3,
            testloader=test_loader,
            path=savepath,
        )
        # for kkz in [10, 20, 50, 100, 300, 500, 999]:
        #     model.set_ka(kkz)
        #     model.set_kk(kkz)
        #     trr.train(
        #         trainloader=train_loader,
        #         valloader=val_loader,
        #         max_epochs=3,
        #         lnr=1e-3,
        #         header="mnist",
        #         testloader=test_loader,
        #     )
