# Transparent Mini-Med 🩺

**Lightweight, Explainable AI for Medical Image Diagnostics**

> ⚠️ **Educational/Research Use Only** — Not clinically validated. Not a substitute for medical advice.

---

## Overview

Transparent Mini-Med is a lightweight, explainable deep learning system for detecting **pneumonia** from chest X-ray images. It uses a **ResNet18** backbone (~11.2M params) and integrates **Grad-CAM** to produce heatmaps that visually explain *why* the model predicted a given diagnosis — bridging the gap between AI accuracy and clinical trust.

For a detailed technical breakdown, see our [System Overview](docs/system_overview.md) and [Methodology](docs/methodology.md) docs.

```
Input X-ray → MiniMedNet → Diagnosis (Pneumonia / Normal)
                         ↘ Grad-CAM → Visual Heatmap Explanation
```

---

## Features

- 🧠 **ResNet18 Backbone** — high-accuracy feature extraction using a pre-trained medical-ready architecture
- 🔥 **Grad-CAM** heatmaps — visual explanations highlighting influential image regions
- 🖥️ **Flask Web UI** — premium dark-mode dashboard for uploading X-rays and viewing results without refresh
- 📓 **Google Colab Notebook** — end-to-end training, evaluation, and demo
- 📊 **Full Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- 🧪 **Unit Tests** — pytest suite covering model, Grad-CAM, and inference pipeline

---

## Project Structure

The project uses a flat, professional structure:

```
dnn/
├── app.py                      # Main entry point (Flask Backend)
├── src/                        # Core AI Engine (Architectures, Inference, Explainability)
│   ├── architectures/mini_med_net.py
│   ├── training/               # Dataset, training loop, evaluation
│   ├── explainability/         # Grad-CAM & overlay logic
│   ├── inference/predictor.py  # End-to-end prediction engine
│   └── utils/image_utils.py    # Image processing
├── static/                     # UI Assets (CSS, JS, Uploads)
├── templates/                  # UI Layouts (HTML)
├── scripts/                    # Utility scripts (Predict, Demo Gen, etc.)
├── tests/                      # Unit tests
├── config/                     # YAML configurations
├── models/                     # Saved weights (demo or trained)
├── data/                       # Dataset samples & Diagnostic history
└── docs/                       # Technical documentation
    ├── methodology.md
    └── system_overview.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Generate Demo Weights (Optional)
If you don't have a trained model yet, generate dummy weights for testing the pipeline:
```powershell
python scripts/generate_demo_model.py
```

### 3. Launch the Clinical UI (Browser)
To run the project in your browser:
```powershell
python app.py
```
*Click the link shown in the terminal to open the dashboard (usually http://localhost:5000).*

### 4. Run Diagnosis (CLI)
Alternatively, you can run diagnosis directly in the terminal:
```powershell
python scripts/predict.py --image data/samples/sample_xray.jpeg
```

---

## Training Your Own Model

### a) Download Dataset

Download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```
Extract to `data/chest_xrays/` so the structure is:
```
data/chest_xrays/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
```

### b) Train

```bash
# Windows
scripts\run_training.bat

# Linux / macOS
bash scripts/run_training.sh
```

Or directly:
```bash
python core/training/train.py --config config/training_config.yaml
```

### c) Evaluate

```bash
python core/training/evaluate.py --weights models/mini_med_net_best.pth --data-dir data/chest_xrays/test
```

### d) Run Tests

```bash
python -m pytest tests/ -v --cov=core
```

---

## Grad-CAM Explanation

Grad-CAM generates a heatmap highlighting which regions of the X-ray most influenced the model's diagnosis:

1. A forward pass computes the class probability
2. Gradients flow back to the last residual block (`layer4`)
3. Feature maps are weighted by global-average-pooled gradients
4. ReLU filters negative values; the result is resized to 224×224
5. The heatmap is overlaid on the original image with a jet colormap

**Red/yellow regions** = high influence on the prediction  
**Blue/green regions** = low influence

---

## Ethical Statement

This system is a **clinical decision-support tool**, not an autonomous diagnostic engine:
- Physicians must review all predictions before clinical decisions
- The model is not validated on diverse hospital equipment
- Regulatory approval (FDA/EMA) is required before any clinical deployment
- Misuse for autonomous diagnosis without clinical oversight is strongly discouraged

See the [Technical Methodology](docs/methodology.md) for further implementation details.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Transparent Mini-Med — Explainable AI, Built for Trust.*
