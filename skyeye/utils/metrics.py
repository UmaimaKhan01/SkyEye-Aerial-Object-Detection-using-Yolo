"""
Evaluation metrics for object detection
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .general import LOGGER


def box_iou(box1, box2):
    """
    Calculate IoU between boxes
    
    Args:
        box1 (torch.Tensor): First set of boxes (N, 4)
        box2 (torch.Tensor): Second set of boxes (M, 4)
        
    Returns:
        torch.Tensor: IoU matrix (N, M)
    """
    # Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
    box2 = box2.T
    
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    
    # Intersection area
    inter = (torch.min(b1_x2[:, None], b2_x2) - torch.max(b1_x1[:, None], b2_x1)).clamp(0) * \
            (torch.min(b1_y2[:, None], b2_y2) - torch.max(b1_y1[:, None], b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7
    union = (w1[:, None] * h1[:, None]) + (w2 * h2) - inter + 1e-7
    
    return inter / union  # IoU


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    
    Args:
        box1 (torch.Tensor): First box coordinates
        box2 (torch.Tensor): Second box coordinates
        xywh (bool): If True, boxes are in [x, y, w, h] format, else [x1, y1, x2, y2]
        GIoU (bool): Calculate Generalized IoU
        DIoU (bool): Calculate Distance IoU
        CIoU (bool): Calculate Complete IoU
        eps (float): Small constant to avoid division by zero
        
    Returns:
        torch.Tensor: IoU/GIoU/DIoU/CIoU
    """
    # Convert from center-width to coordinates if needed
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    if GIoU or DIoU or CIoU:
        # Convex hull coordinates
        c_x1, c_y1 = torch.min(b1_x1, b2_x1), torch.min(b1_y1, b2_y1)
        c_x2, c_y2 = torch.max(b1_x2, b2_x2), torch.max(b1_y2, b2_y2)
        
        # Convex hull area
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
        
        if DIoU or CIoU:  # Distance or Complete IoU
            # Center points
            b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
            b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
            
            # Diagonal length of convex hull
            c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
            
            # Centers distance squared
            center_dist = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
            
            if CIoU:  # Complete IoU
                # Calculate aspect ratio term
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                
                # CIoU
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                
                return iou - (center_dist / c_diag + v * alpha)
            
            return iou - center_dist / c_diag  # DIoU
        
        # GIoU
        return iou - (c_area - union) / c_area
    
    return iou  # IoU


def compute_ap(recall, precision):
    """
    Compute the average precision using the VOC07/12 11-point method
    
    Args:
        recall (numpy.ndarray): Recall values
        precision (numpy.ndarray): Precision values
        
    Returns:
        tuple: AP value, interpolated precision, recall
    """
    # Ensure recall array starts with 0 and ends with 1
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute precision envelope (maximum precision for each recall value)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Calculate area under precision-recall curve (AUC)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(''), names=()):
    """
    Compute the average precision per class
    
    Args:
        tp (numpy.ndarray): True positives
        conf (numpy.ndarray): Confidence scores
        pred_cls (numpy.ndarray): Predicted classes
        target_cls (numpy.ndarray): Target classes
        plot (bool): Whether to plot PR curves
        save_dir (Path): Directory to save plots
        names (tuple): Class names for plotting
        
    Returns:
        tuple: Precision, recall, AP, F1, unique classes
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes, counts = np.unique(target_cls, return_counts=True)
    num_classes = unique_classes.shape[0]
    
    # Create tensors for per-class metrics
    ap = np.zeros((num_classes, tp.shape[1]))
    precision = np.zeros((num_classes, 1000))
    recall = np.zeros((num_classes, 1000))
    
    # Compute metrics for each class
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_pred = i.sum()  # Number of predictions
        
        if n_pred == 0 or n_gt == 0:
            continue
        
        # Accumulate false positives and true positives
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        recall[ci] = np.interp(-np.linspace(0, 1, 1000), -conf[i], recall_curve[:, 0])
        
        # Precision
        precision_curve = tpc / (tpc + fpc)
        precision[ci] = np.interp(-np.linspace(0, 1, 1000), -conf[i], precision_curve[:, 0])
        
        # AP from precision-recall curve
        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall_curve[:, j], precision_curve[:, j])
    
    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    
    # Format class names for plotting
    if plot and len(names):
        plot_classes = {i: names[int(c)] for i, c in enumerate(unique_classes)}
        
        # Plot precision-recall curves
        from .visualization import plot_precision_recall_curve
        plot_precision_recall_curve(
            np.linspace(0, 1, 1000), 
            [precision[i] for i in range(num_classes)], 
            ap, 
            save_dir, 
            names=list(plot_classes.values())
        )
    
    # Return metrics for all classes
    i = f1.mean(0).argmax()  # max F1 index
    return precision[:, i], recall[:, i], ap, f1[:, i], unique_classes


class ConfusionMatrix:
    """
    Confusion matrix for object detection
    """
    
    def __init__(self, num_classes, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize a confusion matrix
        
        Args:
            num_classes (int): Number of classes
            conf_threshold (float): Confidence threshold for predictions
            iou_threshold (float): IoU threshold for matching
        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
    
    def process_batch(self, detections, labels):
        """
        Update confusion matrix for one batch
        
        Args:
            detections (torch.Tensor): Predicted bounding boxes (N, 6) [x1, y1, x2, y2, conf, cls]
            labels (torch.Tensor): Ground truth labels (M, 5) [cls, x1, y1, x2, y2]
        """
        # Filter detections by confidence
        detections = detections[detections[:, 4] > self.conf_threshold]
        
        # Extract class information
        gt_classes = labels[:, 0].int().cpu().numpy() if len(labels) else []
        det_classes = detections[:, 5].int().cpu().numpy() if len(detections) else []
        
        # Calculate IoU between predictions and ground truth
        iou = box_iou(labels[:, 1:5], detections[:, :4]) if len(labels) and len(detections) else torch.zeros(0)
        
        # Match predictions to ground truth based on IoU
        matched_indices = torch.nonzero(iou > self.iou_threshold, as_tuple=False)
        
        if matched_indices.shape[0]:
            # Sort matches by IoU
            matched_ious = iou[matched_indices[:, 0], matched_indices[:, 1]]
            matched_indices = matched_indices[matched_ious.argsort(descending=True)]
            
            # Remove duplicate matches (one ground truth can match only one prediction)
            matches, counts = np.unique(matched_indices[:, 0].cpu().numpy(), return_counts=True)
            matched_indices = matched_indices[np.hstack([
                np.zeros(c, dtype=np.bool8) if c > 1 else np.ones(1, dtype=np.bool8) 
                for c in counts
            ])]
            
            # Count matches in confusion matrix
            for gt_idx, det_idx in matched_indices.cpu().numpy():
                gt_class = gt_classes[gt_idx]
                det_class = det_classes[det_idx]
                self.matrix[det_class, gt_class] += 1
            
            # Count false positives (remaining detections)
            if len(detections):
                matched_det_indices = matched_indices[:, 1].cpu().numpy()
                unmatched_det_indices = np.setdiff1d(np.arange(len(detections)), matched_det_indices)
                for det_idx in unmatched_det_indices:
                    det_class = det_classes[det_idx]
                    self.matrix[det_class, self.num_classes] += 1
        else:
            # Count all detections as false positives
            for det_class in det_classes:
                self.matrix[det_class, self.num_classes] += 1
        
        # Count false negatives (unmatched ground truths)
        if len(labels):
            matched_gt_indices = matched_indices[:, 0].cpu().numpy() if len(matched_indices) else []
            unmatched_gt_indices = np.setdiff1d(np.arange(len(labels)), matched_gt_indices)
            for gt_idx in unmatched_gt_indices:
                gt_class = gt_classes[gt_idx]
                self.matrix[self.num_classes, gt_class] += 1
    
    def plot(self, normalize=True, save_dir=Path(''), names=()):
        """
        Plot confusion matrix
        
        Args:
            normalize (bool): Normalize columns
            save_dir (Path): Directory to save the plot
            names (list): Class names
        """
        try:
            import seaborn as sns
            
            # Normalize matrix if requested
            array = self.matrix.copy()
            if normalize:
                array = array / (array.sum(0).reshape(1, -1) + 1e-16)
                array[array < 0.005] = np.nan  # don't annotate small values
            
            # Create figure and set up plot
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sns.set(font_scale=1.0 if self.num_classes < 50 else 0.8)
            
            # Plot heatmap
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning
                heatmap = sns.heatmap(
                    array, 
                    annot=self.num_classes < 30,
                    annot_kws={"size": 8}, 
                    cmap='Blues',
                    fmt='.2f', 
                    square=True,
                    xticklabels=names + ['background FP'] if names else "auto",
                    yticklabels=names + ['background FN'] if names else "auto"
                )
            
            # Set axis labels
            heatmap.set_facecolor((1, 1, 1))
            heatmap.set_xlabel('True')
            heatmap.set_ylabel('Predicted')
            
            # Save plot
            fig.savefig(save_dir / 'confusion_matrix.png', dpi=250)
            plt.close()
            LOGGER.info(f"Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")
            
        except Exception as e:
            LOGGER.warning(f"WARNING: Confusion matrix plot failed: {e}")
    
    def print(self):
        """Print confusion matrix to console"""
        for i in range(self.num_classes + 1):
            LOGGER.info(' '.join(map(str, self.matrix[i])))


def non_max_suppression(
    prediction, 
    conf_threshold=0.25, 
    iou_threshold=0.45, 
    classes=None, 
    agnostic=False, 
    multi_label=False,
    max_detections=300
):
    """
    Perform non-maximum suppression on detection predictions
    
    Args:
        prediction (torch.Tensor): Predictions tensor (batch_size, num_boxes, num_classes+5)
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        classes (list): Filter by class indices
        agnostic (bool): Class-agnostic NMS
        multi_label (bool): Multiple labels per box
        max_detections (int): Maximum number of detections
        
    Returns:
        list: List of detections, on (n,6) tensor per image [x1, y1, x2, y2, conf, cls]
    """
    # Number of classes
    nc = prediction.shape[2] - 5
    
    # Candidates
    xc = prediction[..., 4] > conf_threshold
    
    # Settings
    min_wh, max_wh = 2, 4096  # min and max box width and height
    max_nms_boxes = 30000  # maximum number of boxes to feed to NMS
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # merge boxes for best mAP (adds 0.5ms/img)
    
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for batch_idx, x in enumerate(prediction):  # image index, image predictions
        # Apply confidence threshold
        x = x[xc[batch_idx]]
        
        # If no boxes remain, skip this image
        if not x.shape[0]:
            continue
        
        # Confidence
        if nc > 1:
            if multi_label:
                # Multiple labels per box
                i, j = (x[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
                x = torch.cat((x[i, :5], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                # Best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((x[:, :5], conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        else:
            # Single class, keep all
            conf = x[:, 4:5]
            j = torch.zeros_like(conf)
            x = torch.cat((x[:, :4], conf, j), 1)
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        # If no boxes remain, skip this image
        n = x.shape[0]
        if not n:
            continue
        
        # Sort by confidence and remove excess boxes
        if n > max_nms_boxes:
            x = x[x[:, 4].argsort(descending=True)[:max_nms_boxes]]
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # Perform NMS
        i = torchvision.ops.nms(boxes, scores, iou_threshold)
        if i.shape[0] > max_detections:
            i = i[:max_detections]
        
        # Merge NMS for higher mAP
        if merge and (1 < n < 3000):
            # Update boxes as weighted mean of boxes by IoU overlap
            iou = box_iou(boxes[i], boxes) > iou_threshold  # IoU matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        
        output[batch_idx] = x[i]
    
    return output
