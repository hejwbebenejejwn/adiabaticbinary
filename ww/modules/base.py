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

    def toBin(self):
        assert self._state == "N", "already binary"
        if self.binW:
            self._kknow = torch.clone(self.get_kk()).item()
            self.set_kk(1e5)
        if self.binA:
            self._kanow = torch.clone(self.get_ka()).item()
            self.set_ka(1e5)
        self._state = "B"
        return self._kknow, self._kanow

    def quitBin(self):
        assert self._state == "B", "not binary"
        if self.binW:
            self.set_kk(self._kknow)
        if self.binA:
            self.set_ka(self._kanow)
        self._state = "N"

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass
