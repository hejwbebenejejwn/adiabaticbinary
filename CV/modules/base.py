import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Base(nn.Module, ABC):
    def __init__(self, binW, binA):
        super(Base, self).__init__()
        self.binW = binW
        self.binA = binA
        self._kanow = None
        self._kknow = None
        self._state = "N"

    def set_kk(self, kka):
        if not self.binW:
            raise NotImplementedError
        for mod in self.modules():
            if hasattr(mod, "kk"):
                mod.set_kk(kka)

    def set_ka(self, kka):
        if not self.binA:
            raise NotImplementedError
        for mod in self.modules():
            if hasattr(mod, "ka"):
                mod.set_ka(kka)

    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        met = False
        for mod in self.modules():
            if hasattr(mod, "kk"):
                if not met:
                    met = True
                    kkk = mod.kk.item()
                assert kkk == mod.kk.item(), "kk not aligned"
        return kkk

    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        met = False
        for mod in self.modules():
            if hasattr(mod, "ka"):
                if not met:
                    met = True
                    kka = mod.ka.item()
                assert kka == mod.ka.item(), "ka not aligned"
        return kka

    def toBin(self):
        assert self._state == "N", "already binary"
        if self.binW:
            self._kknow = self.get_kk()
            self.set_kk(1e7)
        if self.binA:
            self._kanow = self.get_ka()
            self.set_ka(1e7)
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
