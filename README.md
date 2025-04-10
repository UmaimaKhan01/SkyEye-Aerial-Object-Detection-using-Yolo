Perfect — here’s the cleaned-up, rebranded README using **SkyEye** as the project name:

---

# SkyEye: Transformer-Aware Object Detection for Aerial Imagery

**SkyEye** is a specialized object detection framework designed for analyzing aerial and drone-captured images. Built on top of YOLOv5 and enhanced with transformer-based modules, it improves detection accuracy for small, crowded, and overlapping objects often found in top-down scenes.

---

## Overview

- Integrates transformer-based components into the detection head  
- Tailored for drone and surveillance-style imagery  
- Supports optional bounding box refinement using prediction fusion  
- Full pipeline includes annotation conversion, training, inference, and evaluation

---

## Workflow

1. **Dataset Preparation**  
   - Organize aerial dataset (e.g., VisDrone-style)  
   - Convert labels using `convert_labels.py` into YOLO format  

2. **Model Training**  
   - Train with transformer-augmented config using `train.py`  

3. **Inference**  
   - Detect objects on new data using `detect.py`  

4. **Evaluation**  
   - Measure accuracy using `val.py` (mAP, precision, recall)  

5. **Prediction Fusion (Optional)**  
   - Apply `box_fusion.py` to merge results from multiple model runs  

---

## Example Commands

**Train the model:**
```bash
python train.py --img 1536 --batch 4 --epochs 80 --data ./data/skyeye.yaml --weights yolov5l.pt --cfg models/skyeye_t.yaml --name skyeye_exp1
```

**Run detection:**
```bash
python detect.py --weights ./weights/skyeye_t.pt --source path/to/images --img 2016 --save-txt
```

**Evaluate performance:**
```bash
python val.py --weights ./weights/skyeye_t.pt --data ./data/skyeye.yaml --img 2016
```

**Optional: Refine predictions with box-level fusion:**
```bash
python box_fusion.py
```

## Notes

SkyEye is optimized for scenarios where traditional object detectors underperform, such as dense vehicle scenes, pedestrian tracking from drones, or surveillance over complex urban areas. The transformer modules embedded in the prediction head help the model better understand spatial relationships and context.

---

Let me know if you'd like this exported as a markdown file or customized further (e.g., with badge icons, dataset credits, or team information).
