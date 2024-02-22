import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Base(nn.Module, ABC):
    def __init__(self, binW, binA):
        super(Base, self).__init__()
        self.binW = binW
        self.binA = binA
        self.kk = None
        self.ka = None
        self._kanow = None
        self._kknow = None
        self._state = "N"

    @abstractmethod
    def set_kk(self, kka):
        pass

    @abstractmethod
    def set_ka(self, kka):
        pass

    @abstractmethod
    def get_kk(self) -> nn.parameter.Parameter:
        pass

    @abstractmethod
    def get_ka(self) -> nn.parameter.Parameter:
        pass

    @abstractmethod
    def toBin(self):
        pass

    @abstractmethod
    def quitBin(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass
