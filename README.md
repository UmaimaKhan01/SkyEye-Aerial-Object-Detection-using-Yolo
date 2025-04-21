# SkyEye: Transformer-Based Object Detection for Aerial Views

**SkyEye** is a high-performance detection system optimized for top-down visual scenes, such as drone footage and overhead surveillance. It integrates multi-scale feature attention and transformer-powered prediction heads to significantly boost detection accuracy, especially for small or overlapping objects.

<p align="center">
  <img width="800" src="https://i.imgur.com/XA2xBPL.png">
</p>

## Highlights

- **Multi-Level Attention Integration**: Enhances feature flow across spatial layers  
- **Transformer-Enhanced Heads**: Refined attention mechanisms for dense object scenes  
- **Aerial Scene Adaptation**: Architecture optimized for wide-angle drone perspectives  
- **Fast and Scalable**: High-speed inference with reduced overhead  
- **User-Friendly Interface**: Modular CLI and API for quick experimentation and deployment

---

## Installation

```bash
# Clone the SkyEye repository
git clone https://github.com/your-org/skyeye.git
cd skyeye

# Install required libraries
pip install -r requirements.txt

# Enable development mode
pip install -e .
```

---

## Quick Usage

### Inference with Pretrained Weights

```python
import torch
from skyeye.core.detector import SkyEyeDetector

# Load model
model = SkyEyeDetector(weights='weights/skyeye_l.pt')

# Run prediction
results = model('path/to/image.jpg')

# Visualize
results.show()

# Save output
results.save('outputs/')
```

---

### Custom Training Example

```bash
python -m skyeye.cli.train --config configs/models/skyeye_s.yaml --data configs/data/drone.yaml --epochs 100 --batch-size 16
```

---

### Model Evaluation

```bash
python -m skyeye.cli.validate --weights weights/skyeye_l.pt --data configs/data/drone.yaml --img-size 1280
```

---



