"""
Attention mechanisms for SkyEye object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """
    Channel Attention Module that focuses on important channel features
    """
    
    def __init__(self, channels, reduction_ratio=16):
        """
        Initialize a channel attention module
        
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for dimensionality reduction
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling branches
        reduced_channels = max(channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through channel attention module
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor: Attention-enhanced feature map
        """
        batch_size, channels = x.size(0), x.size(1)
        
        # Average pooling branch
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        avg_attention = self.shared_mlp(avg_pool).unsqueeze(-1).unsqueeze(-1)
        
        # Max pooling branch
        max_pool = self.max_pool(x).view(batch_size, channels) 
        max_attention = self.shared_mlp(max_pool).unsqueeze(-1).unsqueeze(-1)
        
        # Combine attention branches
        attention = self.sigmoid(avg_attention + max_attention)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module that focuses on important spatial locations
    """
    
    def __init__(self, kernel_size=7):
        """
        Initialize a spatial attention module
        
        Args:
            kernel_size (int): Size of convolutional kernel
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through spatial attention module
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor: Attention-enhanced feature map
        """
        # Generate channel-wise statistics
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate statistics and generate attention map
        attention_map = torch.cat([avg_map, max_map], dim=1)
        attention_map = self.sigmoid(self.conv(attention_map))
        
        return x * attention_map


class CombinedAttention(nn.Module):
    """
    Combined Channel and Spatial Attention Module (CBAM-style)
    """
    
    def __init__(self, channels, reduction_ratio=16):
        """
        Initialize a combined attention module
        
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for channel attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        """
        Forward pass applying sequential channel and spatial attention
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Attention-enhanced feature map
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CrossLayerAttention(nn.Module):
    """
    Cross-Layer Attention Module for feature fusion between different layers
    """
    
    def __init__(self, query_channels, key_channels, value_channels=None, 
                 region_size=2, output_channels=None, heads=4):
        """
        Initialize cross-layer attention module for feature fusion
        
        Args:
            query_channels (int): Number of channels in query feature map
            key_channels (int): Number of channels in key feature map
            value_channels (int, optional): Number of channels in value feature map
            region_size (int): Size of local region for attention calculation
            output_channels (int, optional): Number of output channels
            heads (int): Number of attention heads
        """
        super().__init__()
        
        if value_channels is None:
            value_channels = key_channels
            
        if output_channels is None:
            output_channels = query_channels
            
        self.scale = 1.0 / math.sqrt(query_channels)
        self.heads = heads
        self.region_size = region_size
        self.query_channels = query_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        
        # Projection layers
        self.query_projection = nn.Conv2d(query_channels, query_channels, kernel_size=1)
        self.key_projection = nn.Conv2d(key_channels, key_channels, kernel_size=1)
        self.value_projection = nn.Conv2d(value_channels, value_channels, kernel_size=1)
        self.output_projection = nn.Conv2d(value_channels, output_channels, kernel_size=1)
        
        self.attention_softmax = nn.Softmax(dim=3)
        
    def forward(self, query, key, value=None):
        """
        Forward pass calculating attention between query and key, applying to value
        
        Args:
            query (Tensor): Query tensor for attention mechanism
            key (Tensor): Key tensor for attention mechanism
            value (Tensor, optional): Value tensor to be weighted (defaults to key)
            
        Returns:
            Tensor: Enhanced feature map
        """
        if value is None:
            value = key
            
        batch_size = query.size(0)
        
        # Project query, key, and value
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        
        # Prepare inputs for attention
        query_h, query_w = query.size(2), query.size(3)
        key_h, key_w = key.size(2), key.size(3)
        
        # Reshape query for multi-head attention
        q = query.view(batch_size, self.heads, self.query_channels // self.heads, query_h, query_w)
        
        # Extract local regions from key and value
        patches_k = []
        patches_v = []
        
        # Sample key and value in local regions centered on each query position
        for i in range(self.region_size):
            for j in range(self.region_size):
                # Use interpolation to align key feature map with query
                offset_key = F.interpolate(key, size=(query_h, query_w), mode='bilinear', align_corners=False)
                offset_value = F.interpolate(value, size=(query_h, query_w), mode='bilinear', align_corners=False)
                
                patches_k.append(offset_key)
                patches_v.append(offset_value)
                
        # Stack patches and reshape for attention calculation
        k = torch.stack(patches_k, dim=2)
        v = torch.stack(patches_v, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(batch_size, self.heads, self.key_channels // self.heads, 
                  self.region_size * self.region_size, query_h, query_w)
        v = v.view(batch_size, self.heads, self.value_channels // self.heads, 
                  self.region_size * self.region_size, query_h, query_w)
        
        # Calculate attention scores
        q = q.unsqueeze(3)  # [B, heads, C//heads, 1, H, W]
        attention = (q * k).sum(dim=2) * self.scale  # [B, heads, region_size*region_size, H, W]
        
        # Apply softmax to get attention weights
        attention = self.attention_softmax(attention)
        
        # Apply attention weights to value
        out = (attention.unsqueeze(2) * v).sum(dim=3)  # [B, heads, C//heads, H, W]
        
        # Combine heads and project to output channels
        out = out.view(batch_size, self.value_channels, query_h, query_w)
        out = self.output_projection(out)
        
        return out


class TransformerLayer(nn.Module):
    """
    Standard Transformer encoder layer with self-attention
    """
    
    def __init__(self, dim, num_heads, feedforward_dim=None, dropout=0.1):
        """
        Initialize a transformer layer
        
        Args:
            dim (int): Dimension of the input
            num_heads (int): Number of attention heads
            feedforward_dim (int, optional): Dimension of feedforward network
            dropout (float): Dropout probability
        """
        super().__init__()
        
        if feedforward_dim is None:
            feedforward_dim = dim * 4
            
        # Multi-headed self-attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through transformer layer
        
        Args:
            x (Tensor): Input tensor [B, C, H, W]
            
        Returns:
            Tensor: Transformed feature map
        """
        # Save original shape and flatten spatial dimensions
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # Self-attention block
        x_norm = self.norm1(x_flat)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.dropout(attn_output)
        
        # Feedforward block
        x_norm = self.norm2(x_flat)
        ff_output = self.feedforward(x_norm)
        x_flat = x_flat + ff_output
        
        # Reshape back to original format
        x = x_flat.permute(1, 2, 0).reshape(b, c, h, w)
        
        return x


class WindowedSelfAttention(nn.Module):
    """
    Windowed Self Attention Module with relative position embedding
    """
    
    def __init__(self, dim, window_size, num_heads):
        """
        Initialize windowed self-attention
        
        Args:
            dim (int): Input feature dimension
            window_size (int): Size of attention window
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Define projections for query, key, value
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Define relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Generate pair-wise relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size)
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Initialize bias table with small values
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x, mask=None):
        """
        Forward pass through windowed self-attention
        
        Args:
            x (Tensor): Input tensor of shape [B*num_windows, window_size*window_size, C]
            mask (Tensor, optional): Attention mask
            
        Returns:
            Tensor: Transformed feature map
        """
        B_, N, C = x.shape
        
        # Generate query, key, value
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Softmax and apply attention to values
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x
