import torch
import torch.nn as nn
from abc import ABC,abstractmethod

class Base(nn.Module,ABC):
    def __init__(self, binW, binA):
        super(Base,self).__init__()
        self.binW = binW
        self.binA = binA
    @abstractmethod 
    def set_kk(self, kka):
        pass
    @abstractmethod 
    def set_ka(self, kka):
        pass
    @abstractmethod 
    def get_kk(self):
        pass
    @abstractmethod 
    def get_ka(self):
        pass
    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass