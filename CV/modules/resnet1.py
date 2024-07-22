import torch
import torch.nn as nn
import modules.layers as layers

# import layers
from modules.base import Base

# from base import Base
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(
        self,
        binW,
        binA,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,
    ):
        super().__init__()
        self.binW = binW
        self.binA = binA
        self.binW = binW
        self.binA = binA
        self.conv1 = (
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
            if not binW
            else layers.BinaryConv2D(in_channel, out_channel, 3, stride)
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = (
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
            if not binW
            else layers.BinaryConv2D(out_channel, out_channel)
        )

        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(Base):
    def __init__(self, binW, num_class=10, binA=False):
        super().__init__(binW, binA)
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._resblock(64, 64, 2)
        self.layer2 = self._resblock(64, 128, 2, 2)
        self.layer3 = self._resblock(128, 256, 2, 2)
        self.layer4 = self._resblock(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,num_class)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, layers.BinaryConv2D)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _resblock(self, in_channel, out_channel, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channel != out_channel:
            downsample = (
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channel),
                )
                if not self.binW
                else nn.Sequential(
                    layers.BinaryConv2D(in_channel, out_channel, 1, stride, padding=0),
                    nn.BatchNorm2d(out_channel),
                    layers.BinaryConv2D(in_channel, out_channel, 1, stride, padding=0),
                    nn.BatchNorm2d(out_channel),
                )
            )

        boks = []
        boks.append(
            BasicBlock(self.binW, self.binA, in_channel, out_channel, stride, downsample)
        )
        for _ in range(1, blocks):
            boks.append(
                BasicBlock(
                    self.binW,
                    self.binA,
                    out_channel,
                    out_channel,
                )
            )

        return nn.Sequential(*boks)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


if __name__ == "__main__":
    model = ResNet(0, 0)
    x = torch.rand(3, 3, 32, 32)
    y = model(x)
