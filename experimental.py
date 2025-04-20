# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math
import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download


class CrossProjection(nn.Module):
    # Cross-axis downsampling using asymmetric convolutions
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, groups=1, expand=1.0, residual=False):
        super().__init__()
        mid_channels = int(out_channels * expand)
        self.layer1 = Conv(in_channels, mid_channels, (1, ksize), (1, stride))
        self.layer2 = Conv(mid_channels, out_channels, (ksize, 1), (stride, 1), g=groups)
        self.residual = residual and in_channels == out_channels

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out if self.residual else out


class WeightedMerge(nn.Module):
    # Weighted or unweighted summation of multiple feature maps
    def __init__(self, num_inputs, use_weights=False):
        super().__init__()
        self.use_weights = use_weights
        self.indices = range(num_inputs - 1)
        if use_weights:
            self.weights = nn.Parameter(-torch.arange(1.0, num_inputs) / 2, requires_grad=True)

    def forward(self, inputs):
        output = inputs[0]
        if self.use_weights:
            weight_values = torch.sigmoid(self.weights) * 2
            for idx in self.indices:
                output = output + inputs[idx + 1] * weight_values[idx]
        else:
            for idx in self.indices:
                output = output + inputs[idx + 1]
        return output


class MultiKernelConv(nn.Module):
    # Mixed kernel depthwise convolutions with dynamic channel distribution
    def __init__(self, in_channels, out_channels, kernels=(1, 3), stride=1, equal_channels=True):
        super().__init__()
        num_kernels = len(kernels)

        if equal_channels:
            kernel_map = torch.linspace(0, num_kernels - 1E-6, out_channels).floor()
            ch_splits = [(kernel_map == i).sum().item() for i in range(num_kernels)]
        else:
            b = [out_channels] + [0] * num_kernels
            a = np.eye(num_kernels + 1, num_kernels, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernels) ** 2
            a[0] = 1
            ch_splits = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)

        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels, ch, k, stride, k // 2, groups=math.gcd(in_channels, ch), bias=False)
            for k, ch in zip(kernels, ch_splits)
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        features = [block(x) for block in self.blocks]
        return self.activation(self.bn(torch.cat(features, dim=1)))


class ModelGroup(nn.ModuleList):
    # Ensemble module for combining predictions from multiple models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        outputs = []
        for model in self:
            outputs.append(model(x, augment, profile, visualize)[0])
        return torch.cat(outputs, dim=1), None  # concatenation for NMS-based merging


def load_combined(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    container = ModelGroup()
    weight_list = weights if isinstance(weights, list) else [weights]

    for weight_path in weight_list:
        checkpoint = torch.load(attempt_download(weight_path), map_location=map_location)
        net = checkpoint['ema'] if 'ema' in checkpoint else checkpoint['model']
        net = net.float().eval()
        if fuse:
            net = net.fuse()
        container.append(net)

    for module in container.modules():
        if isinstance(module, (nn.ReLU, nn.SiLU, nn.ReLU6, nn.LeakyReLU, nn.Hardswish, Detect, Model)):
            module.inplace = inplace
            if isinstance(module, Detect) and not isinstance(module.anchor_grid, list):
                module.anchor_grid = [torch.zeros(1)] * module.nl
        elif isinstance(module, Conv):
            module._non_persistent_buffers_set = set()

    if len(container) == 1:
        return container[0]
    else:
        print(f'ModelGroup built using: {weights}')
        for attr in ['names']:
            setattr(container, attr, getattr(container[-1], attr))
        container.stride = container[torch.argmax(torch.tensor([m.stride.max() for m in container])).item()].stride
        return container
