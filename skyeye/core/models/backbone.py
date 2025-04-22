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
        c4 = scaled_channels(base_channels * 8)  # After third downsample
        c5 = scaled_channels(base_channels * 16) # After fourth downsample
        
        # Define network stages
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            # Initial focus block to aggregate pixel information
            FocusBlock(3, c1, kernel_size=3),
            # First downsample
            ConvolutionBlock(c1, c2, 3, stride=2),
            # First feature processing block
            CSPBlock(c2, c2, num_blocks=scaled_depth(3))
        )
        
        # Stage 2: Second level features
        self.stage2 = nn.Sequential(
            # Second downsample
            ConvolutionBlock(c2, c3, 3, stride=2),
            # Second feature processing block
            CSPBlock(c3, c3, num_blocks=scaled_depth(9))
        )
        
        # Stage 3: Third level features with attention
        self.stage3 = nn.Sequential(
            # Third downsample
            ConvolutionBlock(c3, c4, 3, stride=2),
            # Third feature processing block
            CSPBlock(c4, c4, num_blocks=scaled_depth(9)),
            # Add attention mechanism to enhance features
            CombinedAttention(c4)
        )
        
        # Stage 4: Fourth level features with SPP
        self.stage4 = nn.Sequential(
            # Fourth downsample
            ConvolutionBlock(c4, c5, 3, stride=2),
            # Feature processing with spatial pyramid pooling
            CSPBlock(c5, c5, num_blocks=scaled_depth(3)),
            SPPBlock(c5, c5)
        )
        
    def forward(self, x):
        """
        Forward pass through backbone
        
        Args:
            x (Tensor): Input image tensor of shape [B, 3, H, W]
            
        Returns:
            List[Tensor]: Multi-scale feature maps
        """
        # Process through stages
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        
        # Return all feature maps for neck/detection head to use
        return [s2, s3, s4]


class CSPDarknet(Backbone):
    """
    CSPDarknet backbone with configurable depth and width multipliers
    """
    
    def __init__(self, base_channels=64, depth_multiple=1.0, width_multiple=1.0):
        """
        Initialize CSPDarknet backbone
        
        Args:
            base_channels (int): Base channel multiplier for the network
            depth_multiple (float): Depth multiplier for scaling layers
            width_multiple (float): Width multiplier for scaling channels
        """
        super().__init__(base_channels, depth_multiple, width_multiple)


class SkyEyeBackbone(nn.Module):
    """
    Complete SkyEye backbone with enhanced attention mechanisms
    """
    
    def __init__(self, base_channels=64, depth_multiple=1.0, width_multiple=1.0):
        """
        Initialize SkyEye backbone
        
        Args:
            base_channels (int): Base channel multiplier for the network
            depth_multiple (float): Depth multiplier for scaling layers
            width_multiple (float): Width multiplier for scaling channels
        """
        super().__init__()
        
        # Create backbone for feature extraction
        self.backbone = CSPDarknet(base_channels, depth_multiple, width_multiple)
        
        # Channel dimensions for feature maps
        self.channels = [
            max(round(base_channels * 2 * width_multiple), 1),  # P3 channels
            max(round(base_channels * 4 * width_multiple), 1),  # P4 channels
            max(round(base_channels * 8 * width_multiple), 1)   # P5 channels
        ]
        
    def forward(self, x):
        """
        Forward pass through SkyEye backbone
        
        Args:
            x (Tensor): Input image tensor
            
        Returns:
            List[Tensor]: Multi-scale feature maps and channel information
        """
        # Get backbone features
        features = self.backbone(x)
        
        # Return features and channel information
        return features, self.channels
