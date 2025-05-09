"""
Object detection script for SkyEye models
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Add parent directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from skyeye.core.data.dataset import LoadImages, LoadStreams
from skyeye.core.models.detector import load_model
from skyeye.utils.general import (LOGGER, check_img_size, check_requirements, colorstr, 
                                 increment_path, non_max_suppression, print_args, 
                                 scale_boxes, xyxy2xywh)
from skyeye.utils.torch_utils import select_device, time_sync
from skyeye.utils.visualization import Annotator, colors


@torch.no_grad()
def run(
    weights='weights/skyeye_s.pt',     # model path
    source='data/images',              # file/dir/URL/glob/screen/0(webcam)
    img_size=640,                      # inference size (pixels)
    conf_thres=0.25,                   # confidence threshold
    iou_thres=0.45,                    # NMS IOU threshold
    max_det=1000,                      # maximum detections per image
    device='',                         # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,                    # show results
    save_txt=False,                    # save results to *.txt
    save_conf=False,                   # save confidences in --save-txt labels
    save_crop=False,                   # save cropped prediction boxes
    nosave=False,                      # do not save images/videos
    classes=None,                      # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,                # class-agnostic NMS
    augment=False,                     # augmented inference
    visualize=False,                   # visualize features
    project='runs/detect',             # save results to project/name
    name='exp',                        # save results to project/name
    exist_ok=False,                    # existing project/name ok, do not increment
    line_thickness=3,                  # bounding box thickness (pixels)
    hide_labels=False,                 # hide labels
    hide_conf=False,                   # hide confidences
    half=False,                        # use FP16 half-precision inference
):
    """
    Run inference on images, videos, directories, streams, etc.
    
    Args:
        weights (str): Model weights path
        source (str): Source directory, file, URL, glob, screen or 0 for webcam
        img_size (int): Image size
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold
        max_det (int): Maximum detections per image
        device (str): Device to use (cuda device or cpu)
        view_img (bool): Show results
        save_txt (bool): Save results to *.txt
        save_conf (bool): Save confidences in --save-txt labels
        save_crop (bool): Save cropped prediction boxes
        nosave (bool): Do not save images/videos
        classes (list): Filter by class
        agnostic_nms (bool): Class-agnostic NMS
        augment (bool): Augmented inference
        visualize (bool): Visualize features
        project (str): Save results to project/name
        name (str): Save results to project/name
        exist_ok (bool): Existing project/name ok, do not increment
        line_thickness (int): Bounding box thickness (pixels)
        hide_labels (bool): Hide labels
        hide_conf (bool): Hide confidences
        half (bool): Use FP16 half-precision inference
    """
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    
    # Determine if source is webcam, file, or directory
    is_file = Path(source).suffix[1:] in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'dng', 
                                         'webp', 'mpo', 'mp4', 'mov', 'avi', 'mkv')
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = load_model(weights, device=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    if half:
        model.half()  # to FP16
    
    # Set image size
    img_size = check_img_size(img_size, s=stride)  # check image size
    
    # Dataloader
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, stride=stride, auto=True)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=img_size, stride=stride, auto=True)
        bs = 1  # batch_size
    
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    model.warmup(imgsz=(1 if model.pt else bs, 3, img_size, img_size))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            # Save cropped detection
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, img_size, img_size)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    return save_dir


def parse_opt():
    """
    Parse command line arguments for detection
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/skyeye_s.pt', help='model path')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    """
    Main function
    
    Args:
        opt (argparse.Namespace): Command line arguments
    """
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
