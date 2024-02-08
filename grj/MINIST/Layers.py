import math

import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


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