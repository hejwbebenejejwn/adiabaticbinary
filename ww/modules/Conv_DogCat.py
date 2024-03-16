import torch
import torch.nn as nn
# import modules.layers as layers
import layers
# from modules.base import Base
from base import Base
import layers

class CatDog(Base):
    def __init__(self, binW, binA):
        super().__init__(binW, binA)
        self.conv1=layers.BinaryConv2D(3,16,3,1,False) if (binW and not binA) else nn.Conv2d(3,16,3, padding = 1,bias=False)
        self.conv2=layers.BinaryConv2D(16,32,3,1,False) if binW else nn.Conv2d(16,32,3, padding=1,bias=False)
        self.conv3=layers.BinaryConv2D(32,64,3,1,False) if binW else nn.Conv2d(32,64,3, padding=1,bias=False)
        self.dense=layers.BinaryDense(50176,2) if (binW and not binA) else nn.Linear(50176,2)
        self.actv1=layers.BinaryActivation() if binA else nn.ReLU()
        self.actv2=layers.BinaryActivation() if binA else nn.ReLU()
        self.actv3=layers.BinaryActivationRL() if (binA and not binW) else nn.ReLU()
        self.pool1=nn.MaxPool2d(2)
        self.pool2=nn.MaxPool2d(2)
        self.pool3=nn.MaxPool2d(2)
        self.batn1=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn2=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn3=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)

    def forward(self, x:torch.Tensor):
        x=self.actv1(self.batn1(self.pool1(self.conv1(x))))
        x=self.actv2(self.batn2(self.pool2(self.conv2(x))))
        x=self.actv3(self.batn3(self.pool3(self.conv3(x))))
        return self.dense(x.view(-1,50176))
    
    def set_kk(self, kka):
        if not self.binW:
            raise NotImplementedError
        self.conv2.set_kk(kka)
        self.conv3.set_kk(kka)
        if not self.binA:
            self.conv1.set_kk(kka)
            self.dense.set_kk(kka)
    
    def set_ka(self,kka):
        if not self.binA:
            raise NotImplementedError
        self.actv1.set_kk(kka)
        self.actv2.set_kk(kka)
        if not self.binW:
            self.actv3.set_kk(kka)
    
    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        return self.conv2.kk
    
    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        return self.actv1.kk

if __name__=="__main__":
    inputs=torch.randn(4,3,224,224)
    model=CatDog(True,True)
    model.set_kk(1)
    print(model.get_kk().item())
    model.set_ka(1)
    print(model.get_ka().item())
    print(model(inputs))
    model.toBin()
    print(model(inputs))
    model.quitBin()
    