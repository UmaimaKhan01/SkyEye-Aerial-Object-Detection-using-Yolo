"""
Dataset classes for SkyEye object detection
"""

import os
import cv2
import numpy as np
import torch
import glob
import logging
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from multiprocessing.pool import Pool, ThreadPool
from itertools import repeat

from .augmentation import AerialAugmentor, letterbox
from .loaders import img2label_paths

# Supported image formats
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']


def get_hash(paths):
    """
    Calculate a unique hash for a list of file paths
    
    Args:
        paths (list): List of file paths
        
    Returns:
        str: MD5 hash string
    """
    import hashlib
    total_size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    hash_obj = hashlib.md5(str(total_size).encode())
    hash_obj.update(''.join(paths).encode())
    return hash_obj.hexdigest()


def verify_image_label(args):
    """
    Verify an image-label pair
    
    Args:
        args (tuple): Tuple containing (image_file, label_file, prefix)
        
    Returns:
        tuple: Tuple containing verified image and label information
    """
    img_file, label_file, prefix = args
    
    # Default return values
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ''  # missing, found, empty, corrupt, message
    
    try:
        # Verify image
        im = cv2.imread(img_file)
        if im is None:
            nc = 1
            msg = f'{prefix}Image corrupted: {img_file}'
            return [None, None, None, nm, nf, ne, nc, msg]
            
        h, w = im.shape[:2]
        assert (h > 9) and (w > 9), f'Image size too small {h}x{w}'
        
        # Verify labels
        if os.path.isfile(label_file):
            nf = 1  # label found
            with open(label_file, 'r') as f:
                label_data = [x.split() for x in f.read().strip().splitlines() if len(x)]
                
            # Parse label data
            if len(label_data):
                # Convert to numpy array
                l = np.array(label_data, dtype=np.float32)
                
                # Check shape (class, x, y, w, h)
                if len(l):
                    assert l.shape[1] == 5, f'Labels require 5 columns, {l.shape[1]} detected'
                    assert (l >= 0).all(), f'Negative label values found in {img_file}'
                    assert (l[:, 1:] <= 1).all(), f'Non-normalized coordinates found in {img_file}'
                    
                    # Check for duplicate rows
                    if len(np.unique(l, axis=0)) < len(l):
                        l = np.unique(l, axis=0)
                        msg = f'{prefix}WARNING: {img_file}: {len(label_data) - len(l)} duplicate labels removed'
                else:
                    ne = 1  # label empty
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
            
        return [img_file, l, (h, w), nm, nf, ne, nc, msg]
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {img_file}: {e}'
        return [None, None, None, nm, nf, ne, nc, msg]


class AerialDataset(Dataset):
    """
    Dataset for aerial imagery object detection
    Loads images and labels for training and validation
    """
    
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None,
                 rect=False, cache=False, stride=32, pad=0.0, prefix=''):
        """
        Initialize the dataset
        
        Args:
            path (str): Path to dataset directory or list file
            img_size (int): Target image size after preprocessing
            batch_size (int): Batch size for dataloader
            augment (bool): Whether to apply augmentation
            hyp (dict, optional): Hyperparameters for augmentation
            rect (bool): Whether to use rectangular images (minimize padding)
            cache (bool or str): Cache mode ('disk', 'ram', or False)
            stride (int): Stride of the model for appropriate padding
            pad (float): Padding ratio
            prefix (str): Prefix for logging
        """
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = None
        
        # Setup augmentation if needed
        if augment and hyp is not None:
            self.augmentor = AerialAugmentor(hyp)
        else:
            self.augmentor = None
        
        # Find and load images
        self.img_files = []
        p = Path(path)
        
        if p.is_dir():
            # Search for images in directory
            self.img_files.extend(glob.glob(str(p / '**' / '*.*'), recursive=True))
            self.img_files = [x for x in self.img_files if x.split('.')[-1].lower() in IMG_FORMATS]
        elif p.is_file():
            # Read image paths from list file
            with open(p) as f:
                self.img_files = [x.strip() for x in f.read().splitlines() if len(x.strip())]
                
        # Check that images were found
        if not self.img_files:
            raise FileNotFoundError(
