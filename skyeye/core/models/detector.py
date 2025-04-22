"""
Detector models for SkyEye object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
from copy import deepcopy
from pathlib import Path

from .blocks import ConvolutionBlock, CSPBlock, SPPBlock
from .attention import CrossLayerAttention
from .backbone import SkyEyeBackbone


class DetectionHead(nn.Module):
    """
    Detection head for SkyEye object detector
    Processes feature maps and produces bounding box predictions
    """
    
    def __init__(self, num_classes=80, anchors=None, channels=None):
        """
        Initialize detection head
        
        Args:
            num_classes (int): Number of object classes to detect
            anchors (list): Anchor box dimensions for each detection layer
            channels (list): Input channel counts for each detection layer
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # class probabilities + objectness + 4 box coords
        
        # Set up anchors
        self.anchors = anchors if anchors is not None else [
            [[10, 13], [16, 30], [33, 23]],  # small objects
            [[30, 61], [62, 45], [59, 119]],  # medium objects
            [[116, 90], [156, 198], [373, 326]]  # large objects
        ]
        self.num_anchors = len(self.anchors[0]) if self.anchors else 3
        self.num_layers = len(self.anchors) if self.anchors else 3
        
        # Define anchor grids and detection layers
        self.grid = [torch.zeros(1)] * self.num_layers
        self.anchor_grid = [torch.zeros(1)] * self.num_layers
        
        # Input channels for each detection layer if not provided
        if channels is None:
            channels = [256, 512, 1024]
            
        # Create output prediction layers
        self.detection_layers = nn.ModuleList(
            nn.Conv2d(ch, self.num_anchors * self.num_outputs, 1) 
            for ch in channels
        )
        
    def forward(self, feature_maps):
        """
        Forward pass through detection head
        
        Args:
            feature_maps (list): List of feature maps from backbone/neck
            
        Returns:
            list: List of predictions or processed detections
        """
        outputs = []  # Outputs from each layer
        
        # Process each feature map with corresponding detection layer
        for i, (feature_map, detection_layer) in enumerate(zip(feature_maps, self.detection_layers)):
            batch_size, _, grid_h, grid_w = feature_map.shape
            
            # Run detection layer on feature map
            x = detection_layer(feature_map)
            
            # Reshape output: [B, anchors*outputs, H, W] -> [B, anchors, H, W, outputs]
            x = x.view(batch_size, self.num_anchors, self.num_outputs, grid_h, grid_w)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            
            outputs.append(x)
            
        return outputs
        
    def process_detections(self, outputs, input_shape):
        """
        Process raw detections into bounding boxes (inference mode)
        
        Args:
            outputs (list): List of raw detection outputs
            input_shape (tuple): Shape of the input image (h, w)
            
        Returns:
            torch.Tensor: Processed detections with coordinates and scores
        """
        grids = []
        strides = []
        
        # Calculate grid sizes and strides
        for i, output in enumerate(outputs):
            batch_size, _, grid_h, grid_w, _ = output.shape
            
            # Calculate stride for this layer
            stride_h = input_shape[0] / grid_h
            stride_w = input_shape[1] / grid_w
            stride = max(stride_h, stride_w)
            strides.append(torch.tensor(stride, device=output.device))
            
            # Create grid
            yv, xv = torch.meshgrid([torch.arange(grid_h, device=output.device),
                                     torch.arange(grid_w, device=output.device)])
            grid = torch.stack((xv, yv), 2).view(1, 1, grid_h, grid_w, 2).float()
            grids.append(grid)
            
            # Create anchor grid
            anchor_grid = torch.tensor(self.anchors[i], device=output.device)
            anchor_grid = anchor_grid.view(1, self.num_anchors, 1, 1, 2)
            anchor_grid = anchor_grid * stride
            
            # Store grid and anchor grid
            self.grid[i] = grid
            self.anchor_grid[i] = anchor_grid
            
        # Process predictions from each layer
        all_detections = []
        
        for i, output in enumerate(outputs):
            batch_size = output.shape[0]
            
            # Apply sigmoid to outputs
            output = torch.sigmoid(output)
            
            # Adjust coordinates using grid and anchors
            output[..., 0:2] = (output[..., 0:2] * 2 - 0.5 + self.grid[i]) * strides[i]  # xy
            output[..., 2:4] = (output[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            
            # Reshape: [B, anchors, H, W, outputs] -> [B, anchors*H*W, outputs]
            output = output.view(batch_size, -1, self.num_outputs)
            all_detections.append(output)
            
        # Concatenate all detections
        return torch.cat(all_detections, 1)


class FeatureNeck(nn.Module):
    """
    Feature pyramid network (FPN) neck for SkyEye
    Performs top-down and bottom-up fusion of multi-scale features
    """
    
    def __init__(self, in_channels, width_multiple=1.0):
        """
        Initialize feature neck
        
        Args:
            in_channels (list): List of input channel dimensions for each level
            width_multiple (float): Width multiplier for scaling channels
        """
        super().__init__()
        
        # Scale channel dimensions
        def scaled_channels(x):
            return max(round(x * width_multiple), 1)
        
        # Define channel dimensions for neck processing
        # Each level gets progressively smaller
        c3, c4, c5 = in_channels
        
        # Top-down path (larger stride -> smaller stride)
        # Reduce channels in higher-level features
        self.lateral_conv5 = ConvolutionBlock(c5, scaled_channels(c4), 1, 1)
        self.lateral_conv4 = ConvolutionBlock(c4, scaled_channels(c3), 1, 1)
        
        # Process merged features
        self.fpn_conv4 = CSPBlock(scaled_channels(c4) * 2, scaled_channels(c4), num_blocks=3)
        self.fpn_conv3 = CSPBlock(scaled_channels(c3) * 2, scaled_channels(c3), num_blocks=3)
        
        # Bottom-up path (smaller stride -> larger stride)
        # Increase feature map stride
        self.downsample3 = ConvolutionBlock(scaled_channels(c3), scaled_channels(c3), 3, 2)
        self.downsample4 = ConvolutionBlock(scaled_channels(c4), scaled_channels(c4), 3, 2)
        
        # Process downsampled features
        self.pan_conv4 = CSPBlock(scaled_channels(c3) + scaled_channels(c4), scaled_channels(c4), num_blocks=3)
        self.pan_conv5 = CSPBlock(scaled_channels(c4) + scaled_channels(c5), scaled_channels(c5), num_blocks=3)
        
        # Output channels for each feature level
        self.out_channels = [
            scaled_channels(c3),
            scaled_channels(c4),
            scaled_channels(c5)
        ]
        
    def forward(self, features):
        """
        Forward pass through neck to merge multi-scale features
        
        Args:
            features (list): List of [P3, P4, P5] features from backbone
            
        Returns:
            list: List of processed feature maps
        """
        p3, p4, p5 = features
        
        # Top-down path
        p5_td = self.lateral_conv5(p5)
        p4_td = self.lateral_conv4(p4)
        
        # Upsample and merge
        p5_upsampled = F.interpolate(p5_td, size=p4.shape[2:], mode='nearest')
        p4_merged = torch.cat([p5_upsampled, p4], dim=1)
        p4_processed = self.fpn_conv4(p4_merged)
        
        p4_upsampled = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_merged = torch.cat([p4_upsampled, p3], dim=1)
        p3_processed = self.fpn_conv3(p3_merged)
        
        # Bottom-up path
        p3_downsample = self.downsample3(p3_processed)
        p4_cat = torch.cat([p3_downsample, p4_processed], dim=1)
        p4_out = self.pan_conv4(p4_cat)
        
        p4_downsample = self.downsample4(p4_out)
        p5_cat = torch.cat([p4_downsample, p5], dim=1)
        p5_out = self.pan_conv5(p5_cat)
        
        return [p3_processed, p4_out, p5_out]


class SkyEyeDetector(nn.Module):
    """
    SkyEye detector model with enhanced fusion and attention for aerial object detection
    """
    
    def __init__(self, cfg='skyeye_s.yaml', channels=3, num_classes=None, anchors=None):
        """
        Initialize SkyEye detector model
        
        Args:
            cfg (str or dict): Path to model configuration file or configuration dict
            channels (int): Number of input image channels
            num_classes (int, optional): Number of classes to detect
            anchors (list, optional): Custom anchor dimensions
        """
        super().__init__()
        
        # Load model configuration
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            # Load YAML configuration file
            cfg_path = Path(cfg)
            with open(cfg_path, errors='ignore') as f:
                self.cfg = yaml.safe_load(f)
                
        # Update configuration if custom values are provided
        if num_classes and num_classes != self.cfg['nc']:
            self.cfg['nc'] = num_classes
            
        if anchors:
            self.cfg['anchors'] = anchors
            
        # Create backbone, neck, and detection head
        self.backbone = SkyEyeBackbone(
            base_channels=self.cfg.get('base_channels', 64),
            depth_multiple=self.cfg.get('depth_multiple', 1.0),
            width_multiple=self.cfg.get('width_multiple', 1.0)
        )
        
        backbone_features, in_channels = self.backbone(torch.zeros(1, channels, 64, 64))
        
        self.neck = FeatureNeck(
            in_channels, 
            width_multiple=self.cfg.get('width_multiple', 1.0)
        )
        
        self.detection_head = DetectionHead(
            num_classes=self.cfg['nc'],
            anchors=self.cfg.get('anchors', None),
            channels=self.neck.out_channels
        )
        
        # Initialize model parameters
        self._initialize_weights()
        
        # Store model stride for inference
        self.stride = torch.tensor([
            64 // backbone_features[0].shape[2],  # Stride for P3
            64 // backbone_features[1].shape[2],  # Stride for P4
            64 // backbone_features[2].shape[2]   # Stride for P5
        ])
        
        # Store class names
        self.names = [str(i) for i in range(self.cfg['nc'])]
    
    def forward(self, x):
        """
        Forward pass through SkyEye detector
        
        Args:
            x (Tensor): Input image tensor
            
        Returns:
            list: List of detections for each layer or processed boxes
        """
        # Extract features from backbone
        backbone_features, _ = self.backbone(x)
        
        # Process features through neck
        neck_features = self.neck(backbone_features)
        
        # Pass features to detection head
        outputs = self.detection_head(neck_features)
        
        # During inference, process detections into boxes
        if not self.training:
            detections = self.detection_head.process_detections(outputs, x.shape[2:])
            return detections, outputs
            
        return outputs
        
    def _initialize_weights(self):
        """
        Initialize model weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def load_from_pretrained(self, weights_path):
        """
        Load weights from pretrained model
        
        Args:
            weights_path (str): Path to pretrained weights file
        
        Returns:
            self: Returns model instance
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Get state dict from checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model'].float().state_dict()
        else:
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
        # Filter out incompatible keys
        model_state_dict = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                               if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        # Load weights
        self.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} layers from {weights_path}")
        
        return self


def parse_model(model_cfg, in_channels=3):
    """
    Parse model configuration file to create a model
    
    Args:
        model_cfg (str or dict): Path to model configuration file or config dict
        in_channels (int): Number of input channels
        
    Returns:
        tuple: Model configuration dictionary and channel count
    """
    # Load configuration
    if isinstance(model_cfg, str):
        with open(model_cfg, errors='ignore') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = model_cfg
        
    # Get base parameters
    base_channels = cfg.get('base_channels', 64)
    depth_multiple = cfg.get('depth_multiple', 1.0)
    width_multiple = cfg.get('width_multiple', 1.0)
    num_classes = cfg.get('nc', 80)
    
    # Return configuration with computed parameters
    return {
        'base_channels': base_channels,
        'depth_multiple': depth_multiple,
        'width_multiple': width_multiple,
        'nc': num_classes,
        'in_channels': in_channels,
        'anchors': cfg.get('anchors', None)
    }


def construct_model(model_cfg, in_channels=3, num_classes=None, anchors=None):
    """
    Construct a SkyEye model from configuration
    
    Args:
        model_cfg (str or dict): Model configuration file or config dictionary
        in_channels (int): Number of input channels
        num_classes (int, optional): Number of classes to detect
        anchors (list, optional): Custom anchor dimensions
        
    Returns:
        SkyEyeDetector: Constructed model
    """
    cfg = parse_model(model_cfg, in_channels)
    
    # Override params if provided
    if num_classes is not None:
        cfg['nc'] = num_classes
    if anchors is not None:
        cfg['anchors'] = anchors
        
    # Create model
    model = SkyEyeDetector(cfg, in_channels)
    
    return model


class EnhancedSkyEyeDetector(SkyEyeDetector):
    """
    Enhanced SkyEye detector with cross-layer attention and transformer modules
    """
    
    def __init__(self, cfg='skyeye_s.yaml', channels=3, num_classes=None, anchors=None):
        """
        Initialize enhanced SkyEye detector with additional attention mechanisms
        
        Args:
            cfg (str or dict): Path to model configuration file or configuration dict
            channels (int): Number of input image channels
            num_classes (int, optional): Number of classes to detect
            anchors (list, optional): Custom anchor dimensions
        """
        super().__init__(cfg, channels, num_classes, anchors)
        
        # Add cross-layer attention between feature maps
        c3, c4, c5 = self.neck.out_channels
        
        # Create cross-layer attention modules for enhanced feature fusion
        self.cross_attention_p5_p4 = CrossLayerAttention(
            query_channels=c4,
            key_channels=c5,
            region_size=2,
            heads=4
        )
        
        self.cross_attention_p4_p3 = CrossLayerAttention(
            query_channels=c3,
            key_channels=c4,
            region_size=2,
            heads=4
        )
        
    def forward(self, x):
        """
        Forward pass with enhanced cross-layer attention
        
        Args:
            x (Tensor): Input image tensor
            
        Returns:
            list: List of detections for each layer or processed boxes
        """
        # Extract features from backbone
        backbone_features, _ = self.backbone(x)
        
        # Process features through neck
        p3, p4, p5 = self.neck(backbone_features)
        
        # Apply cross-layer attention for enhanced feature fusion
        p4_enhanced = self.cross_attention_p5_p4(p4, p5) + p4
        p3_enhanced = self.cross_attention_p4_p3(p3, p4_enhanced) + p3
        
        neck_features = [p3_enhanced, p4_enhanced, p5]
        
        # Pass features to detection head
        outputs = self.detection_head(neck_features)
        
        # During inference, process detections into boxes
        if not self.training:
            detections = self.detection_head.process_detections(outputs, x.shape[2:])
            return detections, outputs
            
        return outputs
