"""
Loss function implementations for SkyEye object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def smooth_bce(eps=0.1):
    """
    Smooth BCE targets for improved training stability.
    Returns positive and negative label smoothing BCE targets.
    
    Args:
        eps (float): Smoothing epsilon
        
    Returns:
        tuple: (positive_target, negative_target)
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, format='xyxy', iou_type='standard', eps=1e-7):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1 (Tensor): First box coordinates
        box2 (Tensor): Second box coordinates
        format (str): Box format ('xyxy' or 'xywh')
        iou_type (str): IoU calculation type ('standard', 'giou', 'diou', 'ciou')
        eps (float): Small epsilon to avoid division by zero
        
    Returns:
        Tensor: IoU or variation of IoU depending on type
    """
    # Convert from xywh to xyxy if needed
    if format == 'xywh':
        box1_xyxy = torch.cat([
            box1[..., 0:1] - box1[..., 2:3] / 2,  # x1
            box1[..., 1:2] - box1[..., 3:4] / 2,  # y1
            box1[..., 0:1] + box1[..., 2:3] / 2,  # x2
            box1[..., 1:2] + box1[..., 3:4] / 2,  # y2
        ], dim=-1)
        
        box2_xyxy = torch.cat([
            box2[..., 0:1] - box2[..., 2:3] / 2,  # x1
            box2[..., 1:2] - box2[..., 3:4] / 2,  # y1
            box2[..., 0:1] + box2[..., 2:3] / 2,  # x2
            box2[..., 1:2] + box2[..., 3:4] / 2,  # y2
        ], dim=-1)
    else:
        box1_xyxy = box1
        box2_xyxy = box2
    
    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy.unbind(-1)
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    if iou_type == 'standard':
        return iou
    
    # Calculate extra metrics for other IoU variants
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    
    if iou_type == 'giou':
        # Generalized IoU
        c_area = cw * ch + eps  # convex area
        giou = iou - (c_area - union) / c_area
        return giou
    
    if iou_type in ('diou', 'ciou'):
        # Distance IoU or Complete IoU
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        
        # Center distance squared
        rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + 
                (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
        
        if iou_type == 'diou':
            # Distance IoU
            return iou - rho2 / c2
        
        # Complete IoU with aspect ratio term
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        
        # CIoU alpha parameter
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
            
        return iou - (rho2 / c2 + v * alpha)
    
    # Default return standard IoU
    return iou


class FocalLoss(nn.Module):
    """
    Focal Loss with focal parameter gamma to address class imbalance
    """
    
    def __init__(self, gamma=1.5, alpha=0.25, reduction='mean'):
        """
        Initialize focal loss module
        
        Args:
            gamma (float): Focusing parameter, reduces relative loss for well-classified examples
            alpha (float): Weighting factor, balances positive/negative samples
            reduction (str): Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred, true):
        """
        Calculate focal loss between predictions and targets
        
        Args:
            pred (Tensor): Predictions, with shape [B, N]
            true (Tensor): Target values, with shape [B, N]
            
        Returns:
            Tensor: Computed focal loss
        """
        # Use BCEWithLogitsLoss as base
        bce = F.binary_cross_entropy_with_logits(pred, true, reduction='none')
        
        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate modulating factor
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Apply factors to BCE loss
        loss = alpha_factor * modulating_factor * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss with alpha parameter for handling class imbalance
    """
    
    def __init__(self, alpha=0.05, reduction='mean'):
        """
        Initialize BCE loss with alpha parameter
        
        Args:
            alpha (float): Scaling factor for missing label effects
            reduction (str): Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred, true):
        """
        Calculate BCE loss with alpha parameter
        
        Args:
            pred (Tensor): Predictions, with shape [B, N]
            true (Tensor): Target values, with shape [B, N]
            
        Returns:
            Tensor: Computed BCE loss
        """
        loss = self.loss_fn(pred, true)
        
        # Apply alpha weighting
        pred_prob = torch.sigmoid(pred)
        alpha_factor = torch.abs(true - pred_prob)
        modulating_factor = 1.0 - torch.exp((-alpha_factor) / self.alpha)
        loss *= modulating_factor
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class AerialDetectionLoss(nn.Module):
    """
    Specialized loss for aerial object detection with size-aware weighting
    """
    
    def __init__(self, num_classes=80, scales=(0.5, 0.5, 1.0, 2.0), reduction='mean'):
        """
        Initialize aerial detection loss
        
        Args:
            num_classes (int): Number of classes
            scales (tuple): Scale factors for different loss components
            reduction (str): Loss reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.bce_cls = BCEWithLogitsLoss(reduction=reduction)
        self.bce_obj = BCEWithLogitsLoss(reduction=reduction)
        self.scales = scales  # [bbox, objectness, class, small_obj_scale]
        
    def forward(self, predictions, targets, anchors):
        """
        Calculate aerial detection loss
        
        Args:
            predictions (list): Model output predictions
            targets (Tensor): Ground truth boxes and classes
            anchors (Tensor): Anchor boxes
            
        Returns:
            tuple: (total_loss, box_loss, obj_loss, cls_loss)
        """
        # Initialize loss components
        loss_box, loss_obj, loss_cls = torch.zeros(1, device=targets.device), \
                                       torch.zeros(1, device=targets.device), \
                                       torch.zeros(1, device=targets.device)
        
        # Extract prediction layers
        num_layers = len(predictions)
        
        # Process each prediction layer
        for i, pred in enumerate(predictions):
            # Extract batch, anchors, grid dimensions
            batch_size, num_anchors, grid_h, grid_w, num_outputs = pred.shape
            num_targets = targets.shape[0]
            
            # Check if any targets present
            if num_targets > 0:
                # Convert targets to grid coordinates
                target_boxes = targets[:, 2:6] * torch.tensor([grid_w, grid_h, grid_w, grid_h],
                                                            device=targets.device)
                
                # Calculate IoU between targets and anchors
                anchor_shapes = anchors[i] / torch.tensor([grid_w, grid_h], device=targets.device)
                anchor_ious = bbox_iou(target_boxes[:, 2:4], anchor_shapes, format='xywh')
                
                # Assign targets to best matching anchors
                best_anchor_ious, best_anchor_indices = anchor_ious.max(dim=1)
                
                # Extract target attributes
                target_boxes = target_boxes[best_anchor_ious > 0.2]
                target_classes = targets[best_anchor_ious > 0.2, 1].long()
                
                # Convert target boxes to grid cell coordinates
                gxy = target_boxes[:, 0:2].long()
                gxi, gyi = gxy[:, 0], gxy[:, 1]
                
                # Extract corresponding predictions
                pred_boxes = pred[0, :, gyi, gxi, :4]
                pred_obj = pred[0, :, gyi, gxi, 4]
                pred_cls = pred[0, :, gyi, gxi, 5:]
                
                # Calculate box loss (CIoU)
                box_iou = bbox_iou(pred_boxes, target_boxes, iou_type='ciou')
                loss_box += (1.0 - box_iou).mean() * self.scales[0]
                
                # Calculate objectness loss
                obj_targets = torch.ones_like(pred_obj)
                loss_obj += self.bce_obj(pred_obj, obj_targets) * self.scales[1]
                
                # Calculate class loss
                if self.num_classes > 1:
                    cls_targets = F.one_hot(target_classes, self.num_classes).float()
                    loss_cls += self.bce_cls(pred_cls, cls_targets) * self.scales[2]
                
                # Apply small object scaling based on object area
                small_obj_mask = (target_boxes[:, 2] * target_boxes[:, 3]) < (64 * 64 / (grid_w * grid_h))
                if small_obj_mask.any():
                    loss_box += ((1.0 - box_iou[small_obj_mask]).mean() * self.scales[3])
            
            # Process background/no-match grid cells
            if num_targets == 0 or best_anchor_ious.shape[0] == 0:
                # All cells treated as background
                obj_targets = torch.zeros_like(pred[..., 4])
                loss_obj += self.bce_obj(pred[..., 4], obj_targets).mean() * self.scales[1]
        
        # Combine all loss components
        total_loss = loss_box + loss_obj + loss_cls
        
        return total_loss, loss_box, loss_obj, loss_cls


class ComputeLoss:
    """
    Compute loss for SkyEye detector
    """
    
    def __init__(self, model):
        """
        Initialize loss computation class
        
        Args:
            model: SkyEye detection model
        """
        super().__init__()
        device = next(model.parameters()).device
        
        # Get model hyperparameters
        det = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect() module
        self.det = det
        
        # Hyperparameters
        self.hyp = {
            'box': 0.05,  # box loss gain
            'cls': 0.5,   # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 1.0,   # obj loss gain
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'fl_gamma': 1.5,  # focal loss gamma
            'label_smoothing': 0.0  # label smoothing epsilon
        }
        
        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf
        self.cp, self.cn = smooth_bce(eps=self.hyp.get('label_smoothing', 0.0))

        # Focal loss
        g = self.hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            self.BCEcls = FocalLoss(g, alpha=0.25)
            self.BCEobj = FocalLoss(g, alpha=0.25)

        # Detect() module parameters
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.stride = det.stride
        self.nc = det.nc  # number of classes
        self.nl = det.nl  # number of detection layers
        self.anchors = det.anchors
        
    def __call__(self, predictions, targets, augment=False):
        """
        Compute loss between model predictions and targets
        
        Args:
            predictions (list): List of model predictions
            targets (Tensor): Ground truth boxes and classes [img_idx, class_idx, x, y, w, h]
            augment (bool): Whether predictions come from augmented inference
            
        Returns:
            tuple: (total_loss, box_loss, obj_loss, cls_loss)
        """
        device = targets.device
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        
        # Build targets for compute_loss()
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # Losses
        for i, pi in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            num_targets = b.shape[0]  # number of targets
            if num_targets:
                # Extract target subset corresponding to matched predictions
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                
                # Calculate IoU between prediction and target
                iou = bbox_iou(pbox, tbox[i], format='xywh', iou_type='ciou')
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                score_iou = torch.clamp(iou.detach(), 0)
                tobj[b, a, gj, gi] = score_iou  # iou ratio
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(num_targets), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
            
            # Objectness loss
            lobj += self.BCEobj(pi[..., 4], tobj) * self.balance[i]  # obj loss
            
        # Apply loss weights
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        
        # Return loss components
        loss = lbox + lobj + lcls
        return loss, torch.cat((lbox, lobj, lcls)).detach()
    
    def build_targets(self, predictions, targets):
        """
        Build targets for compute_loss()
        
        Args:
            predictions (list): Model predictions
            targets (Tensor): Ground truth boxes and classes
            
        Returns:
            tuple: (tcls, tbox, indices, anchors)
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        
        # Normalized to gridspace gain
        gain = torch.ones(7, device=targets.device)
        
        # Same as .repeat_interleave(nt)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        
        # Append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        
        # Offsets
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
            ],
            device=targets.device).float() * g
        
        # Process each scale
        for i in range(self.nl):
            anchors, shape = self.anchors[i], predictions[i].shape
            
            # Scale gain for matching boxes
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            
            # Match targets to best anchors
            if nt:
                # Calculate anchor ratios
                r = t[..., 4:6] / anchors[:, None]
                
                # Select best iou/ratio < threshold
                j = torch.max(r, 1. / r).max(2)[0] < 4.0
                t = t[j]  # filter
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                
                # Calculate which offset matches best
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices
            
            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            
        return tcls, tbox, indices, anch
