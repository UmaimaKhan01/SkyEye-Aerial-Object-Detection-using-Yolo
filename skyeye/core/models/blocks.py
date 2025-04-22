"""
Building blocks for SkyEye neural network architecture
"""

import math
import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """Standard convolution block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        """
        Initialize a convolution block with batch normalization and activation
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int, optional): Padding added to both sides of the input. Default is None (auto-padding)
            groups (int): Number of blocked connections from input to output channels
            activation (bool): Whether to use activation function (SiLU)
        """
        super().__init__()
        # Auto-pad to maintain spatial dimensions
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()
        
    def forward(self, x):
        """Forward pass through the convolution block"""
        return self.act(self.bn(self.conv(x)))

    def fused_forward(self, x):
        """Forward pass for fused operations (used after fusing bn into conv weights)"""
        return self.act(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True):
        """
        Initialize a depthwise separable convolution block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            activation (bool): Whether to use activation function
        """
        super().__init__()
        # Use GCD to determine optimal groups (typically set to in_channels for full depthwise)
        groups = math.gcd(in_channels, out_channels)
        self.conv = ConvolutionBlock(in_channels, out_channels, kernel_size, stride, 
                                    groups=groups, activation=activation)
        
    def forward(self, x):
        """Forward pass through the depthwise separable convolution"""
        return self.conv(x)


class BottleneckBlock(nn.Module):
    """Standard bottleneck block with expansion and shortcut connection"""
    
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        """
        Initialize a bottleneck block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            shortcut (bool): Whether to use a shortcut/residual connection
            expansion (float): Channel expansion factor for hidden layers
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvolutionBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvolutionBlock(hidden_channels, out_channels, 3, 1)
        self.use_shortcut = shortcut and in_channels == out_channels
        
    def forward(self, x):
        """Forward pass through the bottleneck block"""
        return x + self.cv2(self.cv1(x)) if self.use_shortcut else self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    """Cross Stage Partial Network (CSP) block with multiple bottlenecks"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, expansion=0.5):
        """
        Initialize a CSP block with multiple bottlenecks
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of bottleneck blocks
            shortcut (bool): Whether to use shortcut connections in bottlenecks
            expansion (float): Channel expansion factor
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvolutionBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvolutionBlock(in_channels, hidden_channels, 1, 1)
        self.cv3 = ConvolutionBlock(2 * hidden_channels, out_channels, 1, 1)
        
        # Create sequential bottleneck modules
        self.bottlenecks = nn.Sequential(*[
            BottleneckBlock(hidden_channels, hidden_channels, shortcut, 1.0) 
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        """Forward pass through the CSP block"""
        y1 = self.bottlenecks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPBlock(nn.Module):
    """Spatial Pyramid Pooling block for multi-scale feature extraction"""
    
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        """
        Initialize a Spatial Pyramid Pooling block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_sizes (tuple): Kernel sizes for the pooling layers
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvolutionBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvolutionBlock(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, 1)
        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes
        ])
        
    def forward(self, x):
        """Forward pass through the SPP block"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [pool(x) for pool in self.pooling], dim=1))


class FocusBlock(nn.Module):
    """Focus block to aggregate information from spatial dimensions into channels"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, activation=True):
        """
        Initialize a Focus block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int, optional): Padding added to both sides of the input
            activation (bool): Whether to use activation function
        """
        super().__init__()
        self.conv = ConvolutionBlock(in_channels * 4, out_channels, kernel_size, stride, padding, activation=activation)
        
    def forward(self, x):
        """
        Forward pass that aggregates information in channel dimension
        Takes input (b,c,h,w) and returns (b,4c,h/2,w/2)
        """
        # Extract patches from 2x2 grid into channels
        patches = [
            x[..., ::2, ::2],      # top-left
            x[..., 1::2, ::2],     # bottom-left
            x[..., ::2, 1::2],     # top-right
            x[..., 1::2, 1::2]     # bottom-right
        ]
        return self.conv(torch.cat(patches, dim=1))
