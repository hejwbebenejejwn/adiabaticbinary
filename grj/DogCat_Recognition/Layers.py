import math
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Optional, Union


class BinaryConv2d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.kk = torch.tensor([999], requires_grad=False)

    def set_kk(self, kknew):
        with torch.no_grad():
            self.kk = self.kk.to(kknew.device)
            self.kk.copy_(kknew)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        # binerize weight
        weight_binerized = torch.tanh(self.weight * self.kk)

        return self._conv_forward(input, weight_binerized, self.bias)

class BinaryLinear(Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.rand(out_features, in_features)*0.2-0.1)
        self.kk = torch.tensor([999], requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # allow external modification of kk
    def set_kk(self, kknew):
        with torch.no_grad():
            self.kk = self.kk.to(kknew.device)
            self.kk.copy_(kknew)

    def forward(self, input):
        # binarize weights
        binarized_weight = torch.tanh(self.weight * self.kk)
        return F.linear(input, binarized_weight, self.bias)