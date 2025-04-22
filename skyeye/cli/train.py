"""
Training script for SkyEye models
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

# Add parent directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from skyeye.core.data.dataset import create_dataloader
from skyeye.core.loss.functions import ComputeLoss
from skyeye.core.models.detector import SkyEyeDetector
from skyeye.utils.download import attempt_download
from skyeye.utils.general import (check_dataset, check_yaml, colorstr, get_latest_run, 
                                  increment_path, init_seeds, labels_to_class_weights, 
                                  print_args, strip_optimizer)
from skyeye.utils.metrics import fitness
from skyeye.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, 
                                     select_device, distributed_zero_first)
from skyeye.cli.validate import validate

# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("skyeye")


def train(hyp, opt, device, callbacks=None):
    """
    Train a SkyEye model
    
    Args:
        hyp (dict): Hyperparameters
        opt (namespace): Command line options
        device (torch.device): Device to use
        callbacks (list): Callback functions
    
    Returns:
        tuple: (final_epoch, best_fitness, best_model)
    """
    # Directories
    save_dir = Path(opt.save_dir)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    last = weights_dir / 'last.pt'
    best = weights_dir / 'best.pt'
    
    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    
    # Configure
    plots = not opt.evolve
    cuda = device.type != 'cpu'
    init_seeds(opt.seed)
    
    # Data
    with distributed_zero_first(opt.local_rank):
        data_dict = check_dataset(opt.data)
    
    train_path, val_path = data_dict['train'], data_dict['val']
    
    # Get class names
    nc = data_dict['nc']  # number of classes
    names = data_dict['names']
    assert len(names) == nc, f"Number of names {len(names)} doesn't match number of classes {nc}"
    
    # Model
    model = SkyEyeDetector(opt.cfg, ch=3, nc=nc).to(device)
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    
    optimizer_params = []
    
    # Parameter groups: 0) biases, 1) weights with decay, 2) weights without decay
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    # Configure optimizer
    if opt.adam:
        optimizer = Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    
    # Add parameter groups with weight decay
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(pg0)} weight (no decay), {len(pg1)} weight, {len(pg2)} bias")
    
    # Scheduler
    if opt.linear_lr:
        # Linear learning rate scheduler
        lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        # Cosine learning rate scheduler
        lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # EMA - Exponential Moving Average of model weights
    ema = ModelEMA(model) if opt.local_rank in [-1, 0] else None
    
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location='cpu')
        
        # Model
        if ckpt['ema']:
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        
        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.epochs < start_epoch:
            LOGGER.info(f"{colorstr('Warning:')} Epochs set to {opt.epochs}, but resuming from epoch {start_epoch}.")
            opt.epochs += start_epoch
            
    # DP mod
