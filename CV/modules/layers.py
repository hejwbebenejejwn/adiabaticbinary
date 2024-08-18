import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryConv2D(nn.Module):
    def __init__(
        self, in_channel, out_channel, ker_size=3, num_stride=1, padding=1, ker_bias=False
    ):
        super(BinaryConv2D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ker_size = ker_size
        self.num_stride = num_stride
        self.padding=padding

        self.weight = nn.Parameter(
            torch.Tensor(out_channel, in_channel, ker_size, ker_size)
        )
        nn.init.uniform_(self.weight, -0.1, 0.1)

        if ker_bias:
            self.ker_bias = nn.Parameter(torch.Tensor(int(out_channel), 1, 1, 1))
            nn.init.constant_(self.ker_bias, 0)
        else:
            self.register_parameter("ker_bias", None)

        self.nmk = nn.Parameter(torch.Tensor([1.0]))
        self.kk = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

    def set_kk(self, kk_new):
        with torch.no_grad():
            self.kk.fill_(kk_new)

    def forward(self, x):
        if self.kk.item() < 1e6:
            weight = (
                torch.tanh(self.weight * self.kk) + self.ker_bias
                if self.ker_bias is not None
                else torch.tanh(self.weight * self.kk)
            )
        else:
            weight = (
                torch.sign(self.weight) + self.ker_bias
                if self.ker_bias is not None
                else torch.sign(self.weight)
            )

        return self.nmk * F.conv2d(x, weight, None, self.num_stride, self.padding)


class BinaryConv2DCL(BinaryConv2D):
    def __init__(
        self, in_channel, out_channel, ker_size=3, num_stride=1, ker_bias=False
    ):
        super().__init__(in_channel, out_channel, ker_size, num_stride, ker_bias)
        self.reg_loss = None

    def forward(self, x):
        if self.kk.item() < 1e6:
            weight = (
                torch.tanh(self.weight * self.kk) + self.ker_bias
                if self.ker_bias is not None
                else torch.tanh(self.weight * self.kk)
            )
        else:
            weight = (
                torch.sign(self.weight) + self.ker_bias
                if self.ker_bias is not None
                else torch.sign(self.weight)
            )
        self.reg_loss = 0.2 * torch.sum(torch.relu(torch.abs(self.weight) - 0.2) ** 2)
        return self.nmk * F.conv2d(x, weight, None, self.num_stride, padding="same")


class BinaryDense(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BinaryDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channel, in_channel))
        self.bias = nn.Parameter(torch.Tensor(out_channel))
        self.nmk = nn.Parameter(torch.Tensor([1.0]))
        self.kk = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.constant_(self.bias, 0.0)

    def set_kk(self, kk_new):
        with torch.no_grad():
            self.kk.fill_(kk_new)

    def forward(self, inputs):
        if self.kk < 1e6:
            return self.nmk * F.linear(
                inputs, torch.tanh(self.weight * self.kk), self.bias
            )
        else:
            return self.nmk * F.linear(inputs, torch.sign(self.weight), self.bias)


class BinaryActivation(nn.Module):
    def __init__(self, ker_bias=False):
        super(BinaryActivation, self).__init__()
        self.ker_bias = ker_bias
        self.bias = nn.Parameter(torch.Tensor([1.0]), requires_grad=ker_bias)
        self.kk = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

    def set_kk(self, kk_new):
        with torch.no_grad():
            self.kk.fill_(kk_new)

    def forward(self, inputs):
        if self.kk < 1e6:
            return torch.tanh(inputs * self.kk) + self.bias
        else:
            return torch.sign(inputs) + self.bias


class BinaryActivationHT(BinaryActivation):
    def __init__(self, ker_bias=False):
        super().__init__(ker_bias)

    def forward(self, inputs):
        if self.kk < 1e6:
            return torch.clip(inputs * self.kk, min=-1, max=1) + self.bias
        else:
            return torch.sign(inputs) + self.bias


class BinaryActivationRL(BinaryActivation):
    def __init__(self, ker_bias=False):
        super().__init__(ker_bias)
        self.bias = nn.Parameter(torch.Tensor([0.0]), requires_grad=ker_bias)

    def forward(self, inputs):
        if self.kk < 1e6:
            return torch.tanh(torch.relu(inputs) * self.kk) + self.bias
        else:
            return torch.sign(inputs) / 2 + (0.5 + self.bias)
