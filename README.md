# PlateScanner

<p align="center">
  <a href="https://github.com/Paulogb98/platescanner/stargazers">
    <img src="https://img.shields.io/github/stars/Paulogb98/platescanner.svg?colorA=orange&colorB=orange&logo=github" alt="GitHub stars">
  </a>
  <a href="https://github.com/Paulogb98/platescanner/issues">
    <img src="https://img.shields.io/github/issues/Paulogb98/platescanner.svg" alt="GitHub issues">
  </a>
  <a href="https://github.com/Paulogb98/platescanner/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Paulogb98/platescanner.svg" alt="GitHub license">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-Latest-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/ONNX-Ready-2D4C8D?style=flat-square&logo=onnx" alt="ONNX" />
  <img src="https://img.shields.io/badge/Docker-CPU%20%26%20GPU-2496ED?style=flat-square&logo=docker" alt="Docker" />
</p>

<p align="center">
  <a href="#about"><strong>About</strong></a> •
  <a href="#features"><strong>Features</strong></a> •
  <a href="#requirements"><strong>Requirements</strong></a> •
  <a href="#installation"><strong>Installation</strong></a> •
  <a href="#usage"><strong>Usage</strong></a> •
  <a href="#performance"><strong>Performance</strong></a> •
  <a href="#contributing"><strong>Contributing</strong></a> •
  <a href="#license"><strong>License</strong></a>
</p>

---

## About

**PlateScanner** is a production-ready system for automatic vehicle license plate detection and character recognition. Using YOLOv5 for detection and ONNX-based OCR for character recognition, it processes images end-to-end: detects plates, extracts regions, recognizes characters, and exports results to CSV.

Deploy on CPU for accessibility or GPU for high-throughput processing with Docker.

---

## Features

- 🎯 **End-to-End Pipeline** - Automatic detection, cropping, and OCR
- 🌍 **Global Support** - Trained on 9,000+ images from 50+ countries
- 🚀 **High Performance** - 300+ plates/second on CPU, faster on GPU
- 📊 **98% Accuracy** - mAP@0.5 of 0.98 on detection task
- 🐳 **Docker Ready** - CPU and GPU variants included
- 🔄 **Batch Processing** - Process multiple images efficiently
- 📁 **CSV Export** - Structured results with confidence scores

---

## Requirements

- **Python**: 3.10 - 3.12 (strongly recommended)
- **Docker & Docker Compose**: Optional but recommended
- **GPU (Optional)**: NVIDIA with CUDA 11.0+ support

### System Resources

| Resource | Minimum | Recommended |
|----------|---------|------------|
| Memory | 512 MB | 2 GB |
| Disk | 2 GB | 4 GB |
| CPU | 1 core | 4+ cores |

---

## Installation

### Option 1: Docker (Recommended)

**CPU Version:**
```bash
git clone https://github.com/Paulogb98/PlateScanner.git
cd PlateScanner

# Download models and prepare images (see Setup below)

docker compose up -d --build platescanner-cpu
```

**GPU Version (NVIDIA):**
```bash
# Install NVIDIA Container Toolkit first:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker compose up -d --build platescanner-gpu
```

### Option 2: Local Python

```bash
git clone https://github.com/Paulogb98/PlateScanner.git
cd PlateScanner

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download models and prepare images (see Setup below)

python src/main.py
```

---

## Setup Guide

### Step 1: Download Pre-trained Models

Models are available on [Google Drive](https://drive.google.com/drive/folders/1zDDIpQyH9DM2lh3fPqDKHWzF-RDgG42p?usp=sharing).

**Required files:**

| File | Destination |
|------|------------|
| `yolo_detector_model.pt` | `models/detector/` |
| `config.json` | `models/detector/` |
| `license_plates_ocr_model.onnx` | `models/recognizer/` |
| `license_plates_ocr_config.yaml` | `models/recognizer/` |
| `license_plates_ocr_results.json` | `models/recognizer/` |

### Step 2: Prepare Input Images

Place images in `data/raw/` directory:

**Supported formats:** `.jpg`, `.jpeg`, `.png`

```
data/
├── raw/
│   ├── image1.jpg
│   ├── image2.png
│   └── image3.jpeg
```

### Step 3: Run Application

**Docker:**
```bash
docker compose up -d --build platescanner-cpu
docker compose logs -f
```

**Python:**
```bash
python src/main.py
```

### Step 4: Access Results

Results are available in `data/results/ocr_results.csv`:

```
Image Name,Extracted Value
image1.jpg,ABC1234
image2.jpg,XYZ9876
```

Processed images are in `data/processed/`:
- `detected/` - Images with bounding boxes
- `cropped/` - Extracted plate regions

---

## Usage

### Basic Workflow

1. Place images in `data/raw/`
2. Run the application
3. Check `data/results/ocr_results.csv` for results
4. Review annotated images in `data/processed/`

### Python API

```python
from classes.detector import YOLOv5Inference
from classes.recognizer import ONNXPlateRecognizer

# Detection
detector = YOLOv5Inference("models/detector/yolo_detector_model.pt")
results = detector.infer("path/to/image.jpg")
detector.process_results(results, "image.jpg", output_dir="data/processed")

# Recognition
recognizer = ONNXPlateRecognizer(
    "models/recognizer/license_plates_ocr_model.onnx",
    "models/recognizer/license_plates_ocr_config.yaml"
)
recognizer.process_cropped_images("data/processed/cropped", "data/results")
```

---

## Performance

### Benchmarks

| Hardware | Speed | Throughput |
|----------|-------|-----------|
| CPU (8-core) | ~80ms/image | 12.5 plates/s |
| GPU (RTX 3060) | ~15ms/image | 67 plates/s |
| GPU (RTX 4090) | ~8ms/image | 125 plates/s |

### Model Metrics

| Metric | Value |
|--------|-------|
| Detection mAP@0.5 | 0.98 |
| Character Accuracy | ~93% |
| Supported Formats | 50+ countries |
| Dataset Size | 9,000+ images |

---

## Directory Structure

```
PlateScanner/
├── classes/
│   ├── detector.py          # YOLOv5 detection
│   └── recognizer.py        # ONNX OCR
├── data/
│   ├── raw/                 # Input images
│   ├── processed/
│   │   ├── detected/        # Images with boxes
│   │   └── cropped/         # Plate regions
│   └── results/
│       └── ocr_results.csv  # Results
├── models/
│   ├── detector/
│   │   ├── config.json
│   │   └── yolo_detector_model.pt
│   └── recognizer/
│       ├── license_plates_ocr_config.yaml
│       ├── license_plates_ocr_model.onnx
│       └── license_plates_ocr_results.json
├── src/
│   └── main.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Docker GPU not working

Ensure NVIDIA Container Toolkit is installed:
```bash
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
docker run --rm --gpus all nvidia/cuda:11.0-runtime nvidia-smi
```

### Model files not found

Download from [Google Drive](https://drive.google.com/drive/folders/1zDDIpQyH9DM2lh3fPqDKHWzF-RDgG42p?usp=sharing) and extract to correct directories.

### No detections found

Verify image quality and ensure license plates are clearly visible. Try with sample images first.

### Out of memory

Reduce batch size or process images individually. Consider using CPU version with smaller images.

---

## Fine-tuning (Advanced)

To specialize the detection model for specific plate types:

```bash
yolov5 train \
  --data data.yaml \
  --img 640 \
  --batch 16 \
  --weights models/detector/yolo_detector_model.pt \
  --epochs 10
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'feat: add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional plate format support
- Performance optimizations
- Test datasets
- Documentation improvements
- Web API interface
- Cloud deployment guides

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

**Attribution:** OCR modules adapted from [fast_plate_ocr](https://github.com/ankandrew/fast-plate-ocr).

---

## Contact

📧 **Email:** paulogb98@outlook.com

🔗 **LinkedIn:** https://www.linkedin.com/in/paulo-goiss/

---

## Acknowledgments

- **Ultralytics** - YOLOv5 framework
- **ONNX** - Model interoperability
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning framework
- **fast_plate_ocr** - OCR implementation reference

---

<p align="center">
  <strong>Built with ❤️ using Python, PyTorch & ONNX</strong>
  <br />
  <br />
  <a href="https://github.com/Paulogb98/PlateScanner">🔗 Repository</a> •
  <a href="https://github.com/Paulogb98/PlateScanner/issues">📝 Issues</a> •
  <a href="https://github.com/Paulogb98/PlateScanner/releases">📦 Releases</a>
</p>

<p align="center">
  <strong>PlateScanner v1.0</strong> | ✅ Production Ready
</p>