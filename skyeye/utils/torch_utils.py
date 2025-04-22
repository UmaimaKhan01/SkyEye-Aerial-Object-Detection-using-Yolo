"""
PyTorch utility functions for SkyEye
"""

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..utils.general import LOGGER


@contextmanager
def distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something
    
    Args:
        local_rank (int): Local process rank
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def date_modified(path=__file__):
    """
    Return human-readable file modification date
    
    Args:
        path: Path to file
        
    Returns:
        str: Formatted date string
    """
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def get_git_info(path=Path(__file__).parent):
    """
    Returns git description for directory
    
    Args:
        path: Path to directory
        
    Returns:
        str: Git description or empty string
    """
    try:
        return subprocess.check_output(f'git -C {path} describe --tags --long --always', 
                                      shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except Exception:
        return ''


def select_device(device='', batch_size=None):
    """
    Select appropriate device based on input
    
    Args:
        device (str): Device selection ('', '0', '0,1,2,3', 'cpu')
        batch_size (int, optional): Used to verify batch size is compatible with device count
        
    Returns:
        torch.device: Selected device
    """
    device_info = f'SkyEye {get_git_info() or date_modified()} torch {torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '')
    cpu = device == 'cpu'
    
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
    
    cuda = not cpu and torch.cuda.is_available()
    
    if cuda:
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(device_info) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            device_info += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"
    else:
        device_info += 'CPU\n'
    
    LOGGER.info(device_info)
    return torch.device('cuda:0' if cuda else 'cpu')


def time_sync():
    """
    PyTorch accurate timing
    
    Returns:
        float: Current time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(inputs, ops, n=10, device=None):
    """
    SkyEye model profiler for speed and memory usage
    
    Args:
        inputs: Input tensors
        ops: List of operations/modules to profile
        n (int): Number of iterations for profiling
        device: Device to use
        
    Returns:
        list: List of profile results
    """
    results = []
    device = device or select_device()
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}")

    for x in inputs if isinstance(inputs, list) else [inputs]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m
            
            # Check if operation requires half precision
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            
            # Calculate FLOPs
            try:
                from thop import profile as thop_profile
                flops = thop_profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2
            except Exception:
                flops = 0
                
            # Time forward pass
            t = 0
            try:
                for _ in range(n):
                    t_start = time_sync()
                    y = m(x)
                    t += time_sync() - t_start
                    
                # GPU memory
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                
                # Parameters
                params = sum(p.numel() for p in m.parameters()) if isinstance(m, nn.Module) else 0
                
                t = t * 1000 / n  # Convert to milliseconds
                
                print(f'{params:12}{flops:12.4g}{mem:>14.3f}{t:14.4g}')
                results.append((params, flops, mem, t))
                
            except Exception as e:
                print(f"Error profiling {type(m).__name__}: {e}")
                results.append(None)
                
            torch.cuda.empty_cache()
            
    return results


def is_parallel(model):
    """
    Check if model is of DataParallel or DistributedDataParallel type
    
    Args:
        model: PyTorch model
        
    Returns:
        bool: True if model is parallelized
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """
    De-parallelize a model: returns single-GPU model if model is of DP or DDP type
    
    Args:
        model: PyTorch model
        
    Returns:
        model: Non-parallelized model
    """
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    """
    Initialize model weights
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization is used for Conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.SiLU)):
            m.inplace = True


def model_info(model, verbose=True, img_size=640):
    """
    Prints model information
    
    Args:
        model: Model to summarize
        verbose (bool): If True, print layer by layer information
        img_size (int): Input image size
        
    Returns:
        None
    """
    num_params = sum(x.numel() for x in model.parameters())
    num_gradients = sum(x.numel() for x in model.parameters() if x.requires_grad)
    
    if verbose:
        print(f"{'Layer':>5} {'Name':>40} {'Params':>12} {'Shape':>20} {'Type':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            print(f"{i:5} {name:40} {p.numel():12g} {list(p.shape):20} {p.dtype}")
    
    try:
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), 
                         device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        flops = flops * img_size[0] / stride * img_size[1] / stride
        flops_str = f', {flops:.1f} GFLOPs' 
    except Exception:
        flops_str = ''

    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {num_params:,} parameters, "
                f"{num_gradients:,} gradients{flops_str}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """
    Scales an image with padding constraints
    
    Args:
        img: Input image tensor
        ratio (float): Scaling ratio
        same_shape (bool): Maintain same shape with padding
        gs (int): Grid size (stride) constraint
        
    Returns:
        torch.Tensor: Scaled image
    """
    if ratio == 1.0:
        return img
    
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    
    # Interpolate
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
    
    if not same_shape:
        # Calculate padding to make img shape divisible by gs
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def copy_attr(a, b, include=(), exclude=()):
    """
    Copy attributes from b to a
    
    Args:
        a: Destination object
        b: Source object
        include (tuple): Attributes to include
        exclude (tuple): Attributes to exclude
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """
    Model Exponential Moving Average
    Keep a moving average of model weights for better performance
    """
    
    def __init__(self, model, decay=0.9999, updates=0):
        """
        EMA model initialization
        
        Args:
            model: PyTorch model
            decay (float): EMA decay rate
            updates (int): Number of EMA updates
        """
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        # Decay exponential ramp for smooth startup
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        
        # Don't require gradients
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        """
        Update EMA parameters
        
        Args:
            model: Updated model
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
    
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        Update EMA attributes
        
        Args:
            model: Model to copy attributes from
            include: Attributes to include
            exclude: Attributes to exclude
        """
        copy_attr(self.ema, model, include, exclude)


class EarlyStopping:
    """
    Early stopping to halt training when validation metrics stop improving
    """
    
    def __init__(self, patience=30, verbose=True):
        """
        Initialize early stopping
        
        Args:
            patience (int): Number of epochs to wait after fitness stops improving
            verbose (bool): Print stopping information
        """
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float('inf')
        self.verbose = verbose
        self.possible_stop = False
    
    def __call__(self, epoch, fitness):
        """
        Check if training should stop
        
        Args:
            epoch (int): Current epoch
            fitness (float): Current fitness score
            
        Returns:
            bool: True if training should stop
        """
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        
        if stop and self.verbose:
            LOGGER.info(f'Stopping training early at epoch {epoch} as no improvement observed '
                      f'in last {self.patience} epochs. Best results: epoch {self.best_epoch}')
        
        return stop
