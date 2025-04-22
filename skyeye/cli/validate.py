"""
Validation script for SkyEye models
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from skyeye.core.data.dataset import create_dataloader
from skyeye.core.models.detector import load_model
from skyeye.utils.general import (LOGGER, check_dataset, check_img_size, check_yaml, 
                                 colorstr, increment_path, non_max_suppression, 
                                 print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from skyeye.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from skyeye.utils.visualization import plot_images, plot_val_study
from skyeye.utils.torch_utils import select_device, time_sync


def save_one_txt(pred, save_conf, shape, file):
    """
    Save one detection result to file
    
    Args:
        pred (torch.Tensor): Prediction tensor
        save_conf (bool): Save confidence
        shape (tuple): Image shape
        file (Path): Output file path
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in pred.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(pred, jdict, path, class_map):
    """
    Save one detection result to COCO json format
    
    Args:
        pred (torch.Tensor): Prediction tensor
        jdict (list): JSON dictionary
        path (Path): Image file path
        class_map (dict): Class ID mapping
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(pred[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(pred.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)
        })


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    
    Args:
        detections (tensor): (n, 6) - xyxy, conf, class
        labels (tensor): (m, 5) - class, xyxy
        iouv (tensor): IoU thresholds
        
    Returns:
        (tensor): Correct prediction matrix
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    
    for i in range(len(iouv)):
        # IoU threshold + class criterion
        x = torch.where((iou >= iouv[i]) & correct_class)
        
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detection, iou]
            
            if x[0].shape[0] > 1:
                # Sort by IoU (highest first)
                matches = matches[matches[:, 2].argsort(descending=True)]
                
                # Remove duplicate detections
                matches = matches[torch.unique(matches[:, 1], return_index=True)[1]]
                
                # Remove duplicate labels
                matches = matches[torch.unique(matches[:, 0], return_index=True)[1]]
            
            # Register as correct
            correct[matches[:, 1].long(), i] = True
    
    return correct


@torch.no_grad()
def validate(
    data,
    weights=None,
    batch_size=32,
    img_size=640,
    conf_thres=0.001,
    iou_thres=0.6,
    task='val',
    device='',
    workers=8,
    single_cls=False,
    augment=False,
    verbose=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=False,
    save_json=False,
    project='runs/val',
    name='exp',
    exist_ok=False,
    half=True,
    model=None,
    dataloader=None,
    save_dir=Path(''),
    plots=True,
    compute_loss=None,
):
    """
    Validate a trained model on a dataset
    
    Args:
        data (str): Dataset YAML file path
        weights (str): Model weights file path
        batch_size (int): Batch size
        img_size (int): Image size
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold
        task (str): Task ('val', 'test', 'study')
        device (str): Device to use
        workers (int): Number of dataloader workers
        single_cls (bool): Single class mode
        augment (bool): Augmented validation
        verbose (bool): Verbose output
        save_txt (bool): Save results to text file
        save_hybrid (bool): Save hybrid version of labels
        save_conf (bool): Save confidences in --save-txt labels
        save_json (bool): Save JSON COCO format
        project (str): Project name
        name (str): Run name
        exist_ok (bool): Don't increment run if exists
        half (bool): Use FP16 precision
        model (nn.Module): Model to validate
        dataloader (DataLoader): Dataloader to use
        save_dir (Path): Directory to save results
        plots (bool): Generate plots
        compute_loss (callable): Loss function
        
    Returns:
        tuple: (precision, recall, mAP@0.5, mAP@0.5:0.95, val_loss)
    """
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        
        # Load model
        model = load_model(weights, device=device)
        
        # Get stride and check image size
        stride = int(model.stride.max())
        img_size = check_img_size(img_size, s=stride)
        
        # Data
        data_dict = check_dataset(data)
        is_coco = data_dict.get('val', '').endswith('coco/val2017.txt')
        
    # Half precision
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    
    # Configure
    model.eval()
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Dataloader
    if not training:
        # Run inference once to warm up model
        if device.type != 'cpu':
            model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
        
        # Set up dataloader
        task = task if task in ('train', 'val', 'test') else 'val'  # 'val' as default
        dataloader = create_dataloader(
            data_dict[task], 
            img_size, 
            batch_size,
            stride=stride,
            single_cls=single_cls,
            pad=0.5,
            rect=True,
            workers=workers,
            prefix=colorstr(f'{task}: '))[0]
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    
    # Run inference on validation data
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        batch_size, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        dt[1] += time_sync() - t2
        
        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
        
        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3
        
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            # Predictions
            pred[:, 5] = 0 if single_cls else pred[:, 5]  # force single class if needed
            predn = pred.clone()
            scale_boxes(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            
            # Save
            if save_txt:
                save_one_txt(predn, save_conf, shape, save_dir / 'labels' / (path.stem + '.txt'))
            
            if save_json:
                save_one_json(predn, jdict, path, names)
                
        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            plot_images(img, targets, paths, f, names)
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            plot_images(img, output_to_target(out), paths, f, names)
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    
    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, img_size, img_size)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    
    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = str(Path(data_dict.get('path', '../coco')) / 'annotations/instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.json")
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {
