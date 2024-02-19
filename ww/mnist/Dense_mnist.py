from modules.layers import BinaryDense
from modules.layers import BinaryActivationRL
import torch
import torch.nn as nn
from modules.base import Base


class DenseNet(Base):
    def __init__(self, binW, binA):
        super().__init__(binW,binA)
        self.dense1 = (
            BinaryDense(28 * 28, 128) if binW else nn.Linear(28 * 28, 128)
        )
        self.dense2 = BinaryDense(128, 10) if binW else nn.Linear(128, 10)
        self.actv1 = BinaryActivationRL() if binA else nn.ReLU()
        self.actv2 = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.2)

    def set_kk(self, kka):
        if not self.binW:
            raise NotImplementedError
        self.dense1.set_kk(kka)
        self.dense2.set_kk(kka)

    def set_ka(self, kka):
        if not self.binA:
            raise NotImplementedError
        self.actv1.set_kk(kka)

    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        return self.dense1.kk

    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        return self.actv1.kk

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28 * 28)
        x = self.dense1(x)
        x = self.actv1(x)
        x = self.drop(x)
        return self.actv2(x)


if __name__ == "__main__":
    net = DenseNet(True, True)
    print(net.get_ka())
    print("ka")
    print(net.get_kk())
    print("kk")
    x=torch.randn(4,1,28,28)
    print(net(x))
