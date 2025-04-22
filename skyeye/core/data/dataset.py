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
        # Check that images were found
        if not self.img_files:
            raise FileNotFoundError(f'{prefix}No images found in {path}')
            
        # Sort found image files
        self.img_files = sorted(self.img_files)
        
        # Get corresponding label files
        self.label_files = img2label_paths(self.img_files)
        
        # Check cache contents
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        
        # Try to load cached labels
        try:
            cache_data = np.load(cache_path, allow_pickle=True).item()
            assert cache_data.get('hash') == get_hash(self.label_files + self.img_files)
            # Load cached data
            cache = True
        except:
            # Cache doesn't exist or is invalid, verify images and labels
            cache = False
            
        # Verify and cache labels
        if not cache:
            # This can be slow for large datasets
            logging.info(f'{prefix}Checking and caching {path} images and labels...')
            
            # Verify each image and its label in parallel
            with Pool(8) as pool:
                results = list(pool.map(verify_image_label, 
                                        zip(self.img_files, self.label_files, repeat(prefix))))
            
            # Parse verification results
            valid_indices = []
            msgs = []
            nm, nf, ne, nc = 0, 0, 0, 0  # missing, found, empty, corrupt
            labels = []
            shapes = []
            
            for i, (img_file, label_data, shape, _nm, _nf, _ne, _nc, msg) in enumerate(results):
                nm += _nm
                nf += _nf
                ne += _ne
                nc += _nc
                
                if img_file:
                    valid_indices.append(i)
                    labels.append(label_data)
                    shapes.append(shape)
                    if msg:
                        msgs.append(msg)
                        
            # Report status
            logging.info(f'{prefix}Found {nf} images with labels, {nm} missing, {ne} empty, {nc} corrupt')
            
            # Filter data based on valid indices
            self.img_files = [self.img_files[i] for i in valid_indices]
            self.label_files = [self.label_files[i] for i in valid_indices]
            self.labels = labels
            self.shapes = np.array(shapes, dtype=np.float64)
            
            # Cache data
            logging.info(f'{prefix}Caching labels at {cache_path}')
            cache_data = {'hash': get_hash(self.label_files + self.img_files),
                          'labels': labels,
                          'shapes': shapes,
                          'msgs': msgs}
            np.save(cache_path, cache_data)
        else:
            # Load cached data
            self.labels = cache_data['labels']
            self.shapes = cache_data['shapes']
            
        # Setup rectangular training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            aspect_ratio = s[:, 1] / s[:, 0]  # aspect ratio (h/w)
            indices = aspect_ratio.argsort()
            self.img_files = [self.img_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.shapes = s[indices]
            aspect_ratio = aspect_ratio[indices]
            
            # Set training image shapes
            shapes = [[1, 1]] * batch_size
            for i in range(0, len(aspect_ratio), batch_size):
                aspect_ratio_batch = aspect_ratio[i:i + batch_size]
                min_ratio, max_ratio = aspect_ratio_batch.min(), aspect_ratio_batch.max()
                
                if max_ratio < 1:
                    # All images are wider than they are tall
                    shapes[i:i + batch_size] = [[max_ratio, 1]] * len(aspect_ratio_batch)
                elif min_ratio > 1:
                    # All images are taller than they are wide
                    shapes[i:i + batch_size] = [[1, 1/min_ratio]] * len(aspect_ratio_batch)
                    
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
        
        # Cache images into memory for faster training
        self.imgs = [None] * len(self.img_files)
        if cache == 'ram':
            # Cache in RAM
            logging.info(f'{prefix}Caching images in RAM...')
            for i, f in enumerate(self.img_files):
                self.imgs[i] = cv2.imread(f)
                
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Number of items in dataset
        """
        return len(self.img_files)
    
    def __getitem__(self, index):
        """
        Get a single data item from the dataset
        
        Args:
            index (int): Index of the sample to fetch
            
        Returns:
            tuple: Tuple containing (image, labels, img_file, shape)
        """
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        
        # Resize and augment
        shape = self.batch_shapes[index // 32] if self.rect else self.img_size
        
        # Apply letterboxing
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h/h0, w/w0), pad)  # Original size, ratio, padding
        
        # Get labels
        labels = self.labels[index].copy()
        if len(labels):
            # Normalized xywh to pixel xyxy format
            labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            
        # Apply augmentations
        if self.augment:
            img, labels = self.augmentor(img, labels)
            
        # Convert labels back to normalized xywh
        nl = len(labels)
        if nl:
            labels[:, 1:5] = self.xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])
            
        # Create tensors
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
            
        # Convert image
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes
    
    def load_image(self, index):
        """
        Load an image from the dataset
        
        Args:
            index (int): Index of the image to load
            
        Returns:
            tuple: Tuple containing (image array, original dims, resized dims)
        """
        # Load image if not cached
        if self.imgs[index] is None:
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, f'Image not found: {path}'
            h0, w0 = img.shape[:2]  # Original height and width
            
            # Resize image to target size
            r = self.img_size / max(h0, w0)  # Resize ratio
            if r != 1:  # If dimensions are not equal
                resample_method = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=resample_method)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
            
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for dataloader
        
        Args:
            batch (list): List of tuples from __getitem__
            
        Returns:
            tuple: Batched images, labels, paths, and shapes
        """
        img, label, path, shapes = zip(*batch)  # Transposed batch
        
        # Add sample index to targets
        for i, l in enumerate(label):
            l[:, 0] = i  # Add target image index for build_targets()
            
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
    @staticmethod
    def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
        """
        Convert nx4 normalized boxes from [x, y, w, h] to [x1, y1, x2, y2]
        
        Args:
            x (ndarray): Normalized bounding boxes
            w (int): Width
            h (int): Height
            padw (int): Padding width
            padh (int): Padding height
            
        Returns:
            ndarray: Converted bounding boxes
        """
        # Convert normalized xywh to xyxy format
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # Top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # Top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # Bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # Bottom right y
        return y
    
    @staticmethod
    def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
        """
        Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized
        
        Args:
            x (ndarray): Bounding boxes
            w (int): Width
            h (int): Height
            clip (bool): Whether to clip values to [0, 1]
            eps (float): Small epsilon value
            
        Returns:
            ndarray: Normalized bounding boxes
        """
        # Convert xyxy to xywh format and normalize
        y = np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
        y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
        y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
        
        if clip:
            y = np.clip(y, 0.0 - eps, 1.0 + eps)
            
        return y


def create_dataloader(path, img_size, batch_size, stride, augment=False, hyp=None,
                      cache=False, rect=False, rank=-1, workers=8, image_weights=False, 
                      pad=0.0, prefix=''):
    """
    Create a DataLoader with an AerialDataset
    
    Args:
        path (str): Dataset path
        img_size (int): Image size
        batch_size (int): Batch size
        stride (int): Stride for padding
        augment (bool): Augmentation flag
        hyp (dict, optional): Hyperparameters
        cache (bool or str): Cache flag
        rect (bool): Rectangular batching flag
        rank (int): Distributed rank
        workers (int): Number of workers
        image_weights (bool): Use image weights
        pad (float): Padding value
        prefix (str): Logging prefix
        
    Returns:
        tuple: Dataloader and dataset
    """
    # Create dataset
    dataset = AerialDataset(
        path=path,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache=cache,
        stride=stride,
        pad=pad,
        prefix=prefix
    )

    batch_size = min(batch_size, len(dataset))
    workers = min([os.cpu_count() or 1, batch_size if batch_size > 1 else 0, workers])
    
    # Create loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=not rect,
        pin_memory=True,
        collate_fn=AerialDataset.collate_fn
    )
    
    return dataloader, dataset


def load_dataset(dataset_path, img_size=640, batch_size=16, rect=False, workers=8):
    """
    Load a dataset for testing or validation
    
    Args:
        dataset_path (str): Path to dataset
        img_size (int): Image size
        batch_size (int): Batch size
        rect (bool): Rectangular batching flag
        workers (int): Number of workers
        
    Returns:
        DataLoader: Loaded data loader
    """
    return create_dataloader(
        dataset_path,
        img_size,
        batch_size,
        stride=32,
        augment=False,
        rect=rect,
        workers=workers
    )[0]
