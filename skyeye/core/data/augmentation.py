"""
Augmentation utilities for aerial detection datasets
"""

import math
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps


class AerialAugmentation:
    """
    Augmentation suite specifically designed for aerial/drone imagery
    """
    
    def __init__(self, hyp=None):
        """
        Initialize augmentation parameters
        
        Args:
            hyp (dict, optional): Hyperparameters for augmentation
        """
        self.hyp = {
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 10.0,  # Rotation degrees
            'translate': 0.1,  # Translation fraction
            'scale': 0.5,     # Scale augmentation
            'shear': 2.0,     # Shear augmentation
            'perspective': 0.0,  # Perspective augmentation
            'flipud': 0.5,   # Vertical flip probability
            'fliplr': 0.5,   # Horizontal flip probability
            'mosaic': 1.0,   # Mosaic probability
            'mixup': 0.1,    # Mixup probability
            'copy_paste': 0.1,  # Copy-paste probability
            'auto_augment': 0,  # Auto augment
        }
        
        # Update with provided hyperparameters
        if hyp:
            self.hyp.update(hyp)
    
    def apply_hsv(self, img):
        """
        Apply HSV color-space augmentation
        
        Args:
            img (ndarray): Input image (HWC, BGR)
            
        Returns:
            ndarray: Augmented image
        """
        hgain = self.hyp['hsv_h']
        sgain = self.hyp['hsv_s'] 
        vgain = self.hyp['hsv_v']
        
        if hgain or sgain or vgain:
            # Random gains
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
            
            # Convert to HSV
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8
            
            # Apply augmentation
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            
            # Merge channels
            img_hsv = cv2.merge([
                cv2.LUT(hue, lut_hue),
                cv2.LUT(sat, lut_sat),
                cv2.LUT(val, lut_val)
            ])
            
            # Convert back to BGR
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        
        return img
    
    def apply_geometric_transforms(self, img, labels, segments=None):
        """
        Apply geometric transformations (perspective, rotation, scale, etc.)
        
        Args:
            img (ndarray): Input image
            labels (ndarray): Array of labels (n, 5) where n is number of labels
                              and each label is [class, x, y, w, h]
            segments (list, optional): List of segment polygons
            
        Returns:
            tuple: Transformed image and labels
        """
        height, width = img.shape[:2]
        
        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation
        C[1, 2] = -img.shape[0] / 2  # y translation
        
        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.hyp['perspective'], self.hyp['perspective'])  # x perspective
        P[2, 1] = random.uniform(-self.hyp['perspective'], self.hyp['perspective'])  # y perspective
        
        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.hyp['degrees'], self.hyp['degrees'])
        s = random.uniform(1 - self.hyp['scale'], 1 + self.hyp['scale'])
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.hyp['shear'], self.hyp['shear']) * math.pi / 180)  # x shear
        S[1, 0] = math.tan(random.uniform(-self.hyp['shear'], self.hyp['shear']) * math.pi / 180)  # y shear
        
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.hyp['translate'], 0.5 + self.hyp['translate']) * width  # x translation
        T[1, 2] = random.uniform(0.5 - self.hyp['translate'], 0.5 + self.hyp['translate']) * height  # y translation
        
        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        
        # Apply affine transformation
        if (M != np.eye(3)).any():
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        
        # Transform labels
        n = len(labels)
        if n:
            if segments:
                # Apply to segments first
                new_segments = []
                for segment in segments:
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # transform
                    new_segments.append(xy[:, :2])
                
                # Create new boxes from segments
                new_boxes = np.zeros((n, 4))
                for i, segment in enumerate(new_segments):
                    x, y = segment[:, 0], segment[:, 1]
                    new_boxes[i] = [np.min(x), np.min(y), np.max(x), np.max(y)]
            else:
                # Warp boxes directly
                xy = np.ones((n * 4, 3))
                # x1y1, x2y2, x1y2, x2y1
                boxes = labels[:, 1:5].copy()
                x = boxes[:, [0, 2, 0, 2]]
                y = boxes[:, [1, 3, 3, 1]]
                xy[:, 0] = x.reshape(-1)
                xy[:, 1] = y.reshape(-1)
                xy = xy @ M.T  # transform
                xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine
                
                # Create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new_boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
                
                # Clip
                new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
                new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)
            
            # Filter candidates
            i = self._box_candidates(labels[:, 1:5].T * s, new_boxes.T)
            labels = labels[i]
            labels[:, 1:5] = new_boxes[i]
            if segments:
                segments = [segments[i] for i in range(n) if i in i]
        
        return img, labels
    
    def apply_flip_augmentation(self, img, labels):
        """
        Apply random flips (horizontal/vertical)
        
        Args:
            img (ndarray): Input image
            labels (ndarray): Bounding box labels
            
        Returns:
            tuple: Flipped image and updated labels
        """
        # Vertical flip
        if random.random() < self.hyp['flipud']:
            img = np.flipud(img)
            if len(labels):
                labels[:, 2] = 1 - labels[:, 2]  # y center
        
        # Horizontal flip
        if random.random() < self.hyp['fliplr']:
            img = np.fliplr(img)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]  # x center
        
        return img, labels
    
    def load_mosaic(self, index, dataset):
        """
        Load 4 images and combine them in a mosaic
        
        Args:
            index (int): Index of center image
            dataset: Dataset instance to load images from
            
        Returns:
            tuple: Mosaic image and combined labels
        """
        labels4 = []
        s = self.hyp.get('img_size', 640)
        
        # Center coordinates
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in [-s // 2, -s // 2])
        
        # 3 additional random image indices
        indices = [index] + random.choices(range(len(dataset)), k=3)
        
        # Create mosaic
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)  # base image with padding
        
        for i, idx in enumerate(indices):
            # Load image
            img, _, (h, w) = dataset.load_image(idx)
            
            # Position image in mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Place image in mosaic
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust coordinates
            padw, padh = x1a - x1b, y1a - y1b
            
            # Load labels
            labels = dataset.get_labels(idx).copy()
            if labels.size:
                # Convert normalized xywh to pixel xyxy format
                labels[:, 1:5] = self._xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)
            
            labels4.append(labels)
        
        # Combine labels and clip to boundaries
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
        
        # Apply augmentations
        img4, labels4 = self.apply_geometric_transforms(img4, labels4)
        
        return img4, labels4
    
    def mixup(self, img, labels, img2, labels2):
        """
        Apply mixup augmentation
        
        Args:
            img (ndarray): First image
            labels (ndarray): First image labels
            img2 (ndarray): Second image
            labels2 (ndarray): Second image labels
            
        Returns:
            tuple: Mixed image and combined labels
        """
        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        img = (img * r + img2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        
        return img, labels
    
    def _xywhn2xyxy(self, x, w, h, padw=0, padh=0):
        """
        Convert normalized xywh to pixel xyxy format
        
        Args:
            x (ndarray): Normalized coordinates (n, 4) with xywh format
            w (int): Image width
            h (int): Image height
            padw (int): Padding width
            padh (int): Padding height
            
        Returns:
            ndarray: Pixel coordinates in xyxy format
        """
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        
        return y
    
    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
        """
        Filter out invalid boxes after transformation
        
        Args:
            box1 (ndarray): Pre-transformation box (4, n)
            box2 (ndarray): Post-transformation box (4, n)
            wh_thr (float): Width/height threshold (pixels)
            ar_thr (float): Aspect ratio threshold
            area_thr (float): Area ratio threshold
            
        Returns:
            ndarray: Boolean mask of valid boxes
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        
        # Aspect ratio
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        
        # Return candidates meeting criteria
        return (w2 > wh_thr) & (h2 > wh_thr) & \
               (w2 * h2 / (w1 * h1 + eps) > area_thr) & \
               (ar < ar_thr)


class AlbumentationsWrapper:
    """
    Wrapper for Albumentations library integration
    """
    
    def __init__(self, transform_config=None):
        """
        Initialize albumentations transforms
        
        Args:
            transform_config (dict, optional): Configuration for transforms
        """
        self.transform = None
        
        try:
            import albumentations as A
            
            if transform_config is None:
                # Default configuration
                self.transform = A.Compose([
                    A.Blur(p=0.1),
                    A.MedianBlur(p=0.1),
                    A.ToGray(p=0.1),
                    A.CLAHE(p=0.1),
                    A.RandomBrightnessContrast(p=0.2),
                    A.RandomGamma(p=0.1),
                    A.ImageCompression(quality_lower=75, p=0.1)
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            else:
                # Custom configuration
                transforms = []
                
                if transform_config.get('blur', 0):
                    transforms.append(A.Blur(p=transform_config['blur']))
                    
                if transform_config.get('median_blur', 0):
                    transforms.append(A.MedianBlur(p=transform_config['median_blur']))
                    
                if transform_config.get('to_gray', 0):
                    transforms.append(A.ToGray(p=transform_config['to_gray']))
                    
                if transform_config.get('clahe', 0):
                    transforms.append(A.CLAHE(p=transform_config['clahe']))
                    
                if transform_config.get('brightness_contrast', 0):
                    transforms.append(A.RandomBrightnessContrast(p=transform_config['brightness_contrast']))
                    
                if transform_config.get('gamma', 0):
                    transforms.append(A.RandomGamma(p=transform_config['gamma']))
                    
                if transform_config.get('compression', 0):
                    quality = transform_config.get('compression_quality', 75)
                    transforms.append(A.ImageCompression(quality_lower=quality, p=transform_config['compression']))
                
                self.transform = A.Compose(
                    transforms, 
                    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                )
                
        except ImportError:
            print("Warning: Albumentations library not found. Install with 'pip install albumentations'")
    
    def __call__(self, img, labels=None, p=1.0):
        """
        Apply albumentations transforms
        
        Args:
            img (ndarray): Input image
            labels (ndarray, optional): Labels in [class, x, y, w, h] format
            p (float): Probability of applying transforms
            
        Returns:
            tuple: Transformed image and labels
        """
        if self.transform and random.random() < p:
            # Extract class labels
            class_labels = labels[:, 0] if labels is not None else []
            
            # Extract bounding boxes in YOLO format
            bboxes = labels[:, 1:] if labels is not None else []
            
            # Apply transforms
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            
            # Get transformed outputs
            img = transformed['image']
            
            if labels is not None:
                # Reconstruct labels array
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']
                
                if len(transformed_bboxes) > 0:
                    labels = np.zeros((len(transformed_bboxes), 5))
                    labels[:, 0] = transformed_class_labels
                    labels[:, 1:] = np.array(transformed_bboxes)
                else:
                    labels = np.zeros((0, 5))
        
        return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True, stride=32):
    """
    Resize and pad image while maintaining aspect ratio
    
    Args:
        img (ndarray): Input image
        new_shape (tuple): Target shape (height, width)
        color (tuple): Padding color (B, G, R)
        auto (bool): Use minimum rectangle
        scale_fill (bool): Stretch image if True
        scaleup (bool): Allow scaling up
        stride (int): Stride for rounding dimensions
        
    Returns:
        tuple: Resized and padded image, scaling ratio, padding dimensions
    """
    shape = img.shape[:2]  # current shape (height, width)
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Only scale down, do not scale up
    if not scaleup:
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Calculate dimensions with padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch to new_shape
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)
