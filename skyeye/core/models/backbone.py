"""
Backbone networks for SkyEye object detection
"""

import torch
import torch.nn as nn

from .blocks import ConvolutionBlock, CSPBlock, FocusBlock, SPPBlock
from .attention import CombinedAttention


class Backbone(nn.Module):
    """
    Backbone feature extractor for SkyEye models
    Creates a multi-scale feature pyramid
    """
    
    def __init__(self, base_channels=64, depth_multiple=1.0, width_multiple=1.0):
        """
        Initialize backbone network
        
        Args:
            base_channels (int): Base channel multiplier for the network
            depth_multiple (float): Depth multiplier for scaling layers
            width_multiple (float): Width multiplier for scaling channels
        """
        super().__init__()
        
        # Apply width multiplier to channel dimensions
        def scaled_channels(x):
            return max(round(x * width_multiple), 1)
        
        # Apply depth multiplier to number of layer repetitions
        def scaled_depth(x):
            return max(round(x * depth_multiple), 1)
        
        # Define channel dimensions for each stage
        c1 = scaled_channels(base_channels)      # Initial channels
        c2 = scaled_channels(base_channels * 2)  # After first downsample
        c3 = scaled_channels(base_channels * 4)  # After second downsample
        c4 = scaled_channels(
