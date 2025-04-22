"""
Visualization utilities for SkyEye object detection framework
"""

import math
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageFont

from .general import LOGGER, check_file, is_ascii, is_chinese


def check_font(font_path='Arial.ttf', size=10):
    """
    Check if font exists and download if necessary
    
    Args:
        font_path (str): Font file or name
        size (int): Font size
        
    Returns:
        PIL.ImageFont: PIL TrueType Font
    """
    font_path = Path(font_path)
    
    # Check if font exists locally
    try:
        return ImageFont.truetype(str(font_path) if font_path.exists() else font_path.name, size)
    except Exception:
        # Try to use a system font if available
        try:
            return ImageFont.load_default()
        except:
            LOGGER.warning(f'Font {font_path} not found. Using PIL default font.')
            return None


class ImageAnnotator:
    """
    Class for annotating images with bounding boxes and labels
    """
    
    def __init__(self, image, line_width=None, font_size=None, font='Arial.ttf', 
                 pil=False, example='abc'):
        """
        Initialize annotator for drawing on images
        
        Args:
            image (numpy.ndarray or PIL.Image): Image to annotate
            line_width (int, optional): Line width for annotations
            font_size (int, optional): Font size
            font (str): Font name or file
            pil (bool): Force PIL backend
            example (str): Example characters for font validation
        """
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        
        if self.pil:  # Use PIL
            self.im = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(
                font='Arial.Unicode.ttf' if is_chinese(example) else font,
                size=font_size or max(round(self.im.size[0] * 0.03), 12)
            )
        else:  # Use OpenCV
            self.im = image
            self.line_width = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)
    
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        Add one xyxy box to image with label
        
        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2]
            label (str): Box label
            color (tuple): RGB color for the box
            txt_color (tuple): RGB color for the label text
        """
        if self.pil:
            # PIL drawing
            self.draw.rectangle(box, width=self.line_width, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    [box[0], box[1] - h if outside else box[1],
                     box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1],
                    fill=color
                )
                self.draw.text(
                    (box[0], box[1] - h if outside else box[1]),
                    label, fill=txt_color, font=self.font
                )
        else:
            # OpenCV drawing
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.line_width, lineType=cv2.LINE_AA)
            
            if label:
                tf = max(self.line_width - 1, 1)  # font thickness
                font_scale = self.line_width / 3
                w, h = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, font_scale, txt_color, thickness=tf, lineType=cv2.LINE_AA
                )
    
    def rectangle(self, xy, fill=None, outline=None, width=1):
        """
        Add rectangle to image (PIL-only)
        
        Args:
            xy (list): Rectangle coordinates [x1, y1, x2, y2]
            fill (tuple): RGB fill color
            outline (tuple): RGB outline color
            width (int): Line width
        """
        self.draw.rectangle(xy, fill, outline, width)
    
    def text(self, xy, text, txt_color=(255, 255, 255)):
        """
        Add text to image (PIL-only)
        
        Args:
            xy (tuple): Text position (x, y)
            text (str): Text string
            txt_color (tuple): RGB text color
        """
        w, h = self.font.getsize(text)
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)
    
    def result(self):
        """
        Return annotated image as numpy array
        
        Returns:
            numpy.ndarray: Annotated image
        """
        return np.asarray(self.im)


def plot_one_box(box, img, color=(128, 128, 128), label=None, line_thickness=None):
    """
    Plot one bounding box on image
    
    Args:
        box (list): Bounding box [x1, y1, x2, y2]
        img (numpy.ndarray): Image to draw on
        color (tuple): RGB color tuple
        label (str, optional): Box label
        line_thickness (int, optional): Line thickness
        
    Returns:
        numpy.ndarray: Image with box
    """
    # Plots one bounding box on image
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return img


def plot_images(images, targets=None, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """
    Plot a batch of images and their targets
    
    Args:
        images (torch.Tensor or numpy.ndarray): Batch of images
        targets (torch.Tensor or numpy.ndarray, optional): Targets (cls, xyxy)
        paths (list, optional): Image paths
        fname (str): Output filename
        names (dict, optional): Class names
        max_size (int): Maximum image size
        max_subplots (int): Maximum number of subplots
        
    Returns:
        numpy.ndarray: Plotted image grid
    """
    # Convert tensors to numpy arrays
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # De-normalize if needed
    if np.max(images[0]) <= 1:
        images *= 255
    
    # Set up figure
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # upper bound
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    
    # Create mosaic
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    
    # Fill mosaic with images
    for i, img in enumerate(images):
        if i == bs:  # if we've reached max subplots, break
            break
            
        block_x, block_y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        img = img.transpose(1, 2, 0)  # HWC to CHW
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
    
    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    
    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = ImageAnnotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    
    # Draw image borders and labels
    for i in range(i + 1):
        block_x, block_y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([block_x, block_y, block_x + w, block_y + h], None, (255, 255, 255), width=2)
        
        if paths is not None:
            annotator.text((block_x + 5, block_y + 5 + h), str(Path(paths[i]).name[:40]), (220, 220, 220))
        
        # Draw targets
        if targets is not None:
            image_targets = targets[targets[:, 0] == i]
            boxes = image_targets[:, 2:6].T  # xywh to xyxy
            classes = image_targets[:, 1].astype('int')
            
            # Draw boxes
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                cls_name = names[cls] if names else str(cls)
                label = f'{cls_name}'
                
                color = plt.cm.hsv(cls / len(names)) if names else plt.cm.hsv(cls / 80)
                color = tuple(int(c * 255) for c in color[:3])
                
                annotator.box_label(box, label, color=color)
    
    # Save result
    cv2.imwrite(fname, mosaic)
    return mosaic


def plot_labels(labels, names=(), save_dir=Path('')):
    """
    Plot dataset labels
    
    Args:
        labels (numpy.ndarray): Labels array (n, 5) where n is number of labels
        names (tuple): List of class names
        save_dir (Path): Directory to save the plots
    """
    LOGGER.info('Plotting labels...')
    
    # Create figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    ax = ax.ravel()
    
    # Extract classes and box coordinates
    c = labels[:, 0]  # classes
    boxes = labels[:, 1:].transpose()  # boxes [x, y, w, h]
    
    # Create dataframe for seaborn
    df = pd.DataFrame({
        'x': boxes[0], 
        'y': boxes[1], 
        'width': boxes[2], 
        'height': boxes[3]
    })
    
    # Plot histograms for each feature
    sns.histplot(c, ax=ax[0], kde=False, bins=np.arange(len(names) + 1) - 0.5)
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Class Index')
    
    if len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=8)
    
    # Plot label statistics
    sns.scatterplot(data=df, x='x', y='y', ax=ax[1], s=2)
    ax[1].set_xlabel('Normalized X')
    ax[1].set_ylabel('Normalized Y')
    
    sns.histplot(data=df, x='width', y='height', ax=ax[2], bins=50)
    ax[2].set_xlabel('Normalized Width')
    ax[2].set_ylabel('Normalized Height')
    
    # Plot box aspect ratio
    sns.histplot(data=df, x='width', ax=ax[3], kde=True)
    sns.histplot(data=df, x='height', ax=ax[3], kde=True, color='orange')
    ax[3].set_xlabel('Normalized Dimension')
    ax[3].set_ylabel('Frequency')
    ax[3].legend(['Width', 'Height'])
    
    # Save plots
    fig.suptitle('Label Analysis')
    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    plt.close()
    
    # Create correlation plot
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 5})
    plt.savefig(save_dir / 'labels_correlation.jpg', dpi=200)
    plt.close()
    
    LOGGER.info(f'Label analysis plots saved to {save_dir}')


def plot_results(results_file='results.csv', save_dir=Path(''), skip=1):
    """
    Plot training results from results.csv
    
    Args:
        results_file (str): Results CSV file path
        save_dir (Path): Directory to save plots
        skip (int): Skip first n epochs
    """
    try:
        # Load results from CSV
        data = pd.read_csv(results_file)
        
        # Skip first n epochs for stability
        if skip > 0:
            data = data.iloc[skip:]
        
        # Extract epochs
        epochs = data.index.values + skip if skip > 0 else data.index.values
        
        # Create main figure
        fig, ax = plt.subplots(2, 5, figsize=(16, 8), tight_layout=True)
        ax = ax.ravel()
        
        # Define metrics to plot
        metrics = [
            ('box_loss', 'Box Loss'),
            ('obj_loss', 'Objectness Loss'),
            ('cls_loss', 'Classification Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('mAP_0.5', 'mAP@0.5'),
            ('mAP_0.5:0.95', 'mAP@0.5:0.95'),
            ('val_box_loss', 'Validation Box Loss'),
            ('val_obj_loss', 'Validation Objectness Loss'),
            ('val_cls_loss', 'Validation Classification Loss')
        ]
        
        # Plot each metric
        for i, (key, title) in enumerate(metrics):
            if i >= len(ax):  # Skip if we've run out of axes
                break
            
            if key in data.columns:
                ax[i].plot(epochs, data[key], marker='.', linewidth=2, markersize=8)
                ax[i].set_title(title)
                ax[i].set_xlabel('Epoch')
            else:
                ax[i].remove()  # Remove unused axis
        
        # Save plot
        fig.suptitle('Training Results', fontsize=16)
        plt.savefig(save_dir / 'results.png', dpi=200)
        plt.close()
        
        # Plot learning rate
        if 'lr' in data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, data['lr'], marker='.', linewidth=2, markersize=8)
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.savefig(save_dir / 'lr.png', dpi=200)
            plt.close()
        
        LOGGER.info(f'Results plots saved to {save_dir}')
        
    except Exception as e:
        LOGGER.error(f'Error plotting results: {e}')


def plot_precision_recall_curve(px, py, ap, save_dir=Path(''), names=()):
    """
    Plot precision-recall curve
    
    Args:
        px (numpy.ndarray): Recall points
        py (numpy.ndarray): Precision points
        ap (numpy.ndarray): Average precision values
        save_dir (Path): Directory to save the plot
        names (tuple): Class names
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    
    # Stack precision values for each class
    py = np.stack(py, axis=1)
    
    # Plot per-class curves if less than 21 classes
    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
    else:
        ax.plot(px, py, linewidth=1, color='grey')
    
    # Plot mean precision curve
    ax.plot(px, py.mean(1), linewidth=3, color='blue', 
            label=f'all classes {ap[:, 0].mean():.3f} mAP@0.5')
    
    # Set axis labels and limits
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add legend and save
    plt.legend(loc='upper right')
    plt.savefig(save_dir / 'precision_recall_curve.png', dpi=200)
    plt.close()
