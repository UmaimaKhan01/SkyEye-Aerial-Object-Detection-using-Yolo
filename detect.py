# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Set root directory
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
BASE = Path(os.path.relpath(BASE, Path.cwd()))

# Internal modules
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, increment_path, non_max_suppression, scale_coords, xyxy2xywh,
                           print_args, save_one_box, colorstr)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync, load_classifier


@torch.no_grad()
def detect(weights=BASE / 'yolov5s.pt',
           source=BASE / 'data/images',
           img_size=640,
           conf_thres=0.25,
           iou_thres=0.45,
           device='',
           save_txt=False,
           save_img=True,
           view_img=False,
           save_conf=False,
           save_crop=False,
           max_det=1000,
           classes=None,
           agnostic_nms=False,
           half=False,
           augment=False,
           visualize=False,
           project=BASE / 'runs/detect',
           name='exp',
           exist_ok=False,
           line_width=3,
           hide_labels=False,
           hide_conf=False,
           update=False,
           dnn=False):
    # Handle input
    source = str(source)
    is_webcam = source.isnumeric() or source.lower().startswith(('rtsp', 'http', 'https')) or source.endswith('.txt')
    save_results = not source.endswith('.txt') and save_img

    # Create output directory
    out_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (out_dir / 'labels' if save_txt else out_dir).mkdir(parents=True, exist_ok=True)

    # Device setup
    device = select_device(device)
    half &= device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    model.eval()
    stride = int(model.stride.max())
    class_names = model.names
    if half:
        model.half()

    img_size = check_img_size(img_size, s=stride)

    # Dataloader
    if is_webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        data = LoadStreams(source, img_size=img_size, stride=stride, auto=True)
        batch_size = len(data)
    else:
        data = LoadImages(source, img_size=img_size, stride=stride, auto=True)
        batch_size = 1

    seen, timings = 0, [0.0, 0.0, 0.0]
    for path, img, orig, vid_cap, desc in data:
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()

        # Inference
        pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t3 = time_sync()

        timings[0] += t2 - t1
        timings[1] += t3 - t2
        timings[2] += time_sync() - t3

        for i, det in enumerate(pred):
            seen += 1
            if is_webcam:
                p, im0, frame = path[i], orig[i].copy(), data.count
            else:
                p, im0, frame = path, orig.copy(), getattr(data, 'frame', 0)

            p = Path(p)
            save_path = str(out_dir / p.name)
            label_path = str(out_dir / 'labels' / p.stem) + ('' if data.mode == 'image' else f'_{frame}')
            annot = Annotator(im0, line_width=line_width, example=str(class_names))

            if len(det):
                # Adjust bounding boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *box, conf, cls in reversed(det):
                    if save_txt:
                        norm_coords = (xyxy2xywh(torch.tensor(box).view(1, 4)) / torch.tensor(im0.shape)[[1, 0, 1, 0]]).view(-1).tolist()
                        line = (cls, *norm_coords, conf) if save_conf else (cls, *norm_coords)
                        with open(label_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img or save_crop:
                        class_id = int(cls)
                        tag = None if hide_labels else (class_names[class_id] if hide_conf else f'{class_names[class_id]} {conf:.2f}')
                        annot.box_label(box, tag, color=colors(class_id, True))
                        if save_crop:
                            save_one_box(box, im0.copy(), file=out_dir / 'crops' / class_names[class_id] / f'{p.stem}.jpg', BGR=True)

            im0 = annot.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                cv2.imwrite(save_path, im0)

    # Display results
    stats = tuple(x / seen * 1E3 for x in timings)
    LOGGER.info(f'Speed: %.1fms pre, %.1fms infer, %.1fms NMS per image at {(1, 3, *img_size)}' % stats)
    if save_txt or save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', out_dir)}")
    if update:
        from utils.general import strip_optimizer
        strip_optimizer(weights)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=BASE / 'yolov5s.pt')
    parser.add_argument('--source', type=str, default=BASE / 'data/images')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='')
    parser.add_argument('--save-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default=BASE / 'runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-width', default=3, type=int)
    parser.add_argument('--hide-labels', action='store_true')
    parser.add_argument('--hide-conf', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    print_args(Path(__file__).stem, opt)
    return opt


def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    options = get_args()
    detect(**vars(options))


if __name__ == '__main__':
    main()
