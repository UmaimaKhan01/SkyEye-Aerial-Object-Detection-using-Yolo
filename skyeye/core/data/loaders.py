"""
Data loading utilities for SkyEye detection framework
"""

import os
import glob
import hashlib
import random
import time
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from itertools import repeat
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset, DataLoader, distributed

from ..data.augmentation import letterbox, AerialAugmentation


def create_dataloader(path, img_size, batch_size, stride=32, augment=False, hyp=None, 
                      cache=False, rect=False, rank=-1, workers=8, shuffle=False):
    """
    Create data loader for detection datasets
    
    Args:
        path (str): Path to dataset directory or dataset file list
        img_size (int): Image size
        batch_size (int): Batch size
        stride (int): Stride for grid alignment
        augment (bool): Apply data augmentation
        hyp (dict, optional): Hyperparameters for augmentation
        cache (bool): Cache images to RAM or disk
        rect (bool): Use rectangular batching
        rank (int): Rank for distributed training
        workers (int): Number of worker threads
        shuffle (bool): Shuffle dataset
        
    Returns:
        tuple: Data loader and dataset instance
    """
    # Initialize dataset
    dataset = DroneDataset(
        path=path,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache_images=cache,
        stride=stride
    )

    batch_size = min(batch_size, len(dataset))
    num_workers = min(workers, os.cpu_count() or 1, batch_size if batch_size > 1 else 0)
    
    # Setup sampler for distributed training
    sampler = None
    if rank != -1:
        sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling
        
    # For image-weighted sampling
    if dataset.image_weights:
        loader_class = InfiniteDataLoader
    else:
        loader_class = torch.utils.data.DataLoader
        
    # Create dataloader
    dataloader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=DroneDataset.collate_fn
    )
    
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    Dataloader that reuses workers for better throughput
    
    Used for training with image weights
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize InfiniteDataLoader with same args as standard DataLoader"""
        super().__init__(*args, **kwargs)
        self._RepeatSampler = _RepeatSampler(self.batch_sampler)
        self.batch_sampler = self._RepeatSampler
        self.iterator = super().__iter__()
        
    def __len__(self):
        """Return dataset length"""
        return len(self.batch_sampler.sampler)
        
    def __iter__(self):
        """Create infinite iteration"""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats indefinitely
    """
    
    def __init__(self, sampler):
        """
        Initialize repeat sampler
        
        Args:
            sampler: Original sampler
        """
        self.sampler = sampler
        
    def __iter__(self):
        """Create infinite iteration through the sample"""
        while True:
            yield from iter(self.sampler)


class DroneDataset(Dataset):
    """
    Dataset class for aerial/drone imagery with bounding box annotations
    """
    
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, 
                 rect=False, cache_images=False, stride=32, image_weights=False):
        """
        Initialize drone dataset
        
        Args:
            path (str): Path to dataset (directory or file listing images)
            img_size (int): Target image size
            batch_size (int): Batch size
            augment (bool): Apply data augmentation
            hyp (dict, optional): Hyperparameters
            rect (bool): Enable rectangular training
            cache_images (bool): Cache images to RAM or disk
            stride (int): Stride for grid-alignment
            image_weights (bool): Use image weights for sampling
        """
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        
        # Parse path to load image files
        try:
            self.img_files = self._get_img_files(path)
            self.label_files = self._get_label_files()
            cache_path = self._get_cache_path(path)
            
            # Check cache
            if os.path.exists(cache_path):
                # Load cache file
                cache = torch.load(cache_path)
                # Verify cache hash matches current dataset
                if cache.get('hash') == self._get_hash():
                    self.labels = cache['labels']
                    self.shapes = cache['shapes']
                    self.segments = cache.get('segments', [None] * len(self.shapes))
                    print(f'Loaded dataset cache from {cache_path}')
                    return
            
            # Otherwise process and cache dataset
            self._cache_labels(cache_path)
            
        except Exception as e:
            print(f'Error loading dataset: {e}')
            raise
            
        # Set up augmentations
        self.aerial_augment = AerialAugmentation(hyp)
            
        # Setup rectangular training
        if self.rect:
            self._setup_rectangular_training(batch_size)
            
        # Cache images into memory
        self.imgs = [None] * len(self.img_files)
        self.img_hw0 = [None] * len(self.img_files)  # orig hw
        self.img_hw = [None] * len(self.img_files)   # resized hw
        
        if cache_images:
            self._cache_images()
        
    def __len__(self):
        """Return number of items in dataset"""
        return len(self.img_files)
        
    def __getitem__(self, index):
        """
        Get dataset item
        
        Args:
            index (int): Dataset index
            
        Returns:
            tuple: Image tensor, labels, image path, (original_shape, resized_shape)
        """
        # Use mosaic data augmentation
        if self.mosaic and self.augment:
            img, labels = self.aerial_augment.load_mosaic(index, self)
            shapes = None
            
            # MixUp augmentation
            if random.random() < self.hyp.get('mixup', 0):
                img2, labels2 = self.aerial_augment.load_mosaic(
                    random.randint(0, len(self.labels) - 1), 
                    self
                )
                img, labels = self.aerial_augment.mixup(img, labels, img2, labels2)
                
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            
            labels = self.labels[index].copy()
            
            # Adjust bounding box coordinates for letterbox
            if labels.size:
                # Convert normalized xywh to xyxy
                labels[:, 1:] = self._adjust_bbox_coords(labels[:, 1:], ratio, pad)
            
            # Apply augmentations
            if self.augment:
                img, labels = self.aerial_augment.apply_geometric_transforms(img, labels)
                img, labels = self.aerial_augment.apply_hsv(img)
                img, labels = self.aerial_augment.apply_flip_augmentation(img, labels)
        
        # Convert normalized coordinates back to xywh format for training
        nl = len(labels)
        if nl:
            labels[:, 1:5] = self._xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])
        
        # Convert to tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        return img, labels_out, self.img_files[index], shapes
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for batching
        
        Args:
            batch (list): List of tuples (img, label, path, shapes)
            
        Returns:
            tuple: Batched tensors
        """
        img, label, path, shapes = zip(*batch)  # transposed
        
        # Add sample index to targets
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
            
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
    def load_image(self, index):
        """
        Load image from disk
        
        Args:
            index (int): Image index
            
        Returns:
            tuple: Image array, original shape, resized shape
        """
        # Load cached image if available
        if self.imgs[index] is not None:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]
        
        # Load image
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        
        assert img is not None, f"Failed to load image {path}"
        
        h0, w0 = img.shape[:2]  # original hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    
    def get_labels(self, index):
        """
        Get labels for an image
        
        Args:
            index (int): Image index
            
        Returns:
            ndarray: Labels for the image
        """
        return self.labels[index].copy()
    
    def _get_img_files(self, path):
        """
        Get image files from path
        
        Args:
            path (str): Path to dataset
            
        Returns:
            list: List of image file paths
        """
        try:
            path = Path(path)
            
            if path.is_file():  # text file listing images
                with open(path) as f:
                    img_files = [x.strip() for x in f.read().splitlines() if x.strip()]
            elif path.is_dir():  # directory
                img_files = [str(x) for x in sorted(path.glob('*.*')) if x.suffix.lower() in IMG_FORMATS]
            else:
                raise Exception(f"{path} is not a valid file or directory")
            
            assert img_files, f"No images found in {path}"
            return img_files
            
        except Exception as e:
            raise Exception(f"Error loading images from {path}: {e}")
    
    def _get_label_files(self):
        """
        Get label files corresponding to image files
        
        Returns:
            list: List of label file paths
        """
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in self.img_files]
    
    def _get_cache_path(self, path):
        """
        Get cache file path
        
        Args:
            path (str): Dataset path
            
        Returns:
            str: Cache path
        """
        if isinstance(path, str) and Path(path).is_file():
            return Path(path).with_suffix('.cache')
        else:
            return Path(self.label_files[0]).parent.with_suffix('.cache')
    
    def _get_hash(self):
        """
        Calculate hash of dataset
        
        Returns:
            str: Dataset hash
        """
        return hashlib.md5(str(''.join(self.label_files + self.img_files)).encode()).hexdigest()
    
    def _cache_labels(self, cache_path):
        """
        Cache dataset labels
        
        Args:
            cache_path (str): Path to save cache file
        """
        # Verify all label files
        print(f"Verifying {len(self.img_files)} images and labels...")
        
        # Initialize counters and output lists
        nf, nm, ne, nc, msgs = 0, 0, 0, 0, []  # found, missing, empty, corrupt, messages
        self.labels = []
        self.segments = []
        self.shapes = []
        
        # Verify dataset
        pbar = zip(self.img_files, self.label_files)
        for img_file, label_file in pbar:
            try:
                # Verify image
                im = cv2.imread(img_file)
                if im is None:
                    nc += 1
                    msgs.append(f"Corrupt image: {img_file}")
                    continue
                
                h, w = im.shape[:2]
                self.shapes.append((h, w))
                
                # Verify labels
                if os.path.isfile(label_file):
                    nf += 1
                    with open(label_file) as f:
                        label_data = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    
                    # Check for segments format
                    if any(len(x) > 8 for x in label_data):
                        # Handle segmentation data (polygon coordinates)
                        segments = []
                        for x in label_data:
                            segments.append(np.array(x[1:], dtype=np.float32).reshape(-1, 2))
                        
                        # Convert segments to boxes
                        boxes = np.zeros((len(segments), 5))
                        for i, s in enumerate(segments):
                            x, y = s[:, 0], s[:, 1]
                            boxes[i] = np.array([int(x[0]), 
                                               min(x), min(y), max(x), max(y)])
                        
                        self.segments.append(segments)
                    else:
                        # Regular bbox format
                        l = np.array(label_data, dtype=np.float32)
                        if len(l):
                            assert l.shape[1] == 5, f"Labels require 5 columns, {l.shape[1]} columns detected in {label_file}"
                            assert (l >= 0).all(), f"Negative label values found in {label_file}"
                            assert (l[:, 1:] <= 1).all(), f"Non-normalized or out of bounds coordinates found in {label_file}"
                            
                            # Check for duplicate boxes
                            if len(l) > 1:
                                unique_boxes = np.unique(l, axis=0)
                                if len(unique_boxes) < len(l):
                                    l = unique_boxes
                                    msgs.append(f"Duplicate labels removed from {label_file}")
                            
                            self.labels.append(l)
                            self.segments.append(None)
                            continue
                
                    # Empty label file
                    if len(label_data) == 0:
                        ne += 1
                        msgs.append(f"Empty label file: {label_file}")
                        self.labels.append(np.zeros((0, 5), dtype=np.float32))
                        self.segments.append(None)
                else:
                    # Missing label file
                    nm += 1
                    msgs.append(f"Missing label file: {label_file}")
                    self.labels.append(np.zeros((0, 5), dtype=np.float32))
                    self.segments.append(None)
                
            except Exception as e:
                nc += 1
                msgs.append(f"Error processing {img_file}: {e}")
        
        # Print dataset info
        print(f"Dataset summary: {nf} found, {nm} missing, {ne} empty, {nc} corrupt")
        
        # Save cache file
        cache = {
            'labels': self.labels,
            'shapes': np.array(self.shapes),
            'segments': self.segments,
            'hash': self._get_hash()
        }
        torch.save(cache, cache_path)
        print(f"New cache created: {cache_path}")
    
    def _cache_images(self):
        """Cache images to memory for faster training"""
        print("Caching images...")
        
        # Set up ThreadPool for parallel loading
        pool = ThreadPool(8)
        results = pool.imap(lambda x: self.load_image(x), range(len(self.img_files)))
        
        # Load images
        pbar = enumerate(results)
        for i, (img, hw_orig, hw_resized) in pbar:
            self.imgs[i], self.img_hw0[i], self.img_hw[i] = img, hw_orig, hw_resized
            
        pool.close()
    
    def _setup_rectangular_training(self, batch_size):
        """
        Setup for rectangular training (group by aspect ratio)
        
        Args:
            batch_size (int): Batch size
        """
        # Sort by aspect ratio
        s = np.array(self.shapes, dtype=np.float64)
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        i = ar.argsort()
        
        # Rearrange arrays by aspect ratio
        self.img_files = [self.img_files[i] for i in i]
        self.label_files = [self.label_files[i] for i in i]
        self.labels = [self.labels[i] for i in i]
        self.segments = [self.segments[i] for i in i]
        self.shapes = s[i]
        ar = ar[i]
        
        # Set training image shapes
        shapes = [[1, 1]] * batch_size
        for i in range(0, len(shapes), batch_size):
            ari = ar[i:i+batch_size]
            mini, maxi = ari.min(), ari.max()
            
            # Calculate batch shape
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
                
        # Compute batch indices
        self.batch = np.floor(np.arange(len(self.shapes)) / batch_size).astype(int)  # batch index
        self.num_batches = self.batch[-1] + 1  # number of batches
        
        # Rectangular Training shapes
        self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + 0.5).astype(int) * self.stride
    
    def _adjust_bbox_coords(self, bbox, ratio, pad):
        """
        Adjust bounding box coordinates for letterboxing
        
        Args:
            bbox (ndarray): Normalized xywh coordinates
            ratio (tuple): Width and height ratio
            pad (tuple): Padding width and height
            
        Returns:
            ndarray: Adjusted xyxy coordinates
        """
        # Convert normalized xywh to pixel xyxy
        y = np.copy(bbox)
        y[:, 0] = ratio[0] * bbox[:, 0] + pad[0]  # x_center
        y[:, 1] = ratio[1] * bbox[:, 1] + pad[1]  # y_center
        y[:, 2] = ratio[0] * bbox[:, 2]  # width
        y[:, 3] = ratio[1] * bbox[:, 3]  # height
        
        # Convert xywh to xyxy
        xy = np.zeros_like(y)
        xy[:, 0] = y[:, 0] - y[:, 2] / 2  # x1
        xy[:, 1] = y[:, 1] - y[:, 3] / 2  # y1
        xy[:, 2] = y[:, 0] + y[:, 2] / 2  # x2
        xy[:, 3] = y[:, 1] + y[:, 3] / 2  # y2
        
        return xy
    
    def _xyxy2xywhn(self, x, w=640, h=640, clip=False, eps=0.0):
        """
        Convert xyxy to xywh normalized format
        
        Args:
            x (ndarray): bounding boxes in xyxy format
            w (int): image width
            h (int): image height
            clip (bool): clip boxes outside image
            eps (float): minimum box width and height
            
        Returns:
            ndarray: bounding boxes in xywh normalized format
        """
        if clip:
            x[:, 0] = np.clip(x[:, 0], 0, w - eps)
            x[:, 1] = np.clip(x[:, 1], 0, h - eps)
            x[:, 2] = np.clip(x[:, 2], 0, w - eps)
            x[:, 3] = np.clip(x[:, 3], 0, h - eps)
            
        y = np.zeros_like(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
        y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
        y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
        
        return y


# Image formats
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

# Video formats
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


class LoadImagesAndLabels:
    """Legacy alias for DroneDataset class"""
    def __init__(self, *args, **kwargs):
        print("Warning: LoadImagesAndLabels is deprecated, use DroneDataset instead")
        self.dataset = DroneDataset(*args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self.dataset, name)
