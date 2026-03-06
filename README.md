# 🩺 Transparent Mini-Med — Explainable AI for Chest X-Ray Diagnosis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Flask-3.0+-green?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Research%20Only-red?style=for-the-badge" />
</p>

> ⚠️ **Educational/Research Use Only** — Not clinically validated. Not a substitute for professional medical advice.

---

## 📌 What is Transparent Mini-Med?

**Transparent Mini-Med** is a lightweight, explainable deep learning system that detects **pneumonia from chest X-ray images** and visually explains *why* it made its prediction using **Grad-CAM heatmaps**.

Instead of a black-box prediction, the system outputs:
- ✅ **Diagnosis** — Pneumonia or Normal
- 📊 **Confidence Score** — probability (0–100%)
- 🔥 **Heatmap** — highlighted regions that influenced the decision
- 🖼️ **Overlay Image** — heatmap fused over the original X-ray

This bridges the gap between AI accuracy and clinical interpretability.

```
             ┌──────────────────┐
  X-Ray ───► │   MiniMedNet     │ ───► Diagnosis (Pneumonia / Normal)
  Image      │  (ResNet18 CNN)  │ ───► Confidence Score (%)
             └──────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │   Grad-CAM       │ ───► Heatmap (Which region caused it?)
             │   Explainer      │ ───► Overlay Image (Visual Proof)
             └──────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **ResNet18 Backbone** | Pre-trained CNN with ~11.2M parameters, fine-tuned for chest X-rays |
| 🔥 **Grad-CAM Explainability** | Gradient-weighted class activation maps highlight the diagnosis region |
| 🖥️ **Flask Web Dashboard** | Premium dark-mode UI to upload X-rays and see results live in browser |
| 📁 **Scan History** | Persistent SQLite database of past diagnoses with delete/clear features |
| 📊 **Full Evaluation Suite** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| 🧪 **Unit Test Suite** | Pytest tests covering model architecture, Grad-CAM, and predictor pipeline |
| ⚙️ **CLI Scripts** | Command-line tools for predicting, training, and evaluating |
| 🔒 **Safe & Ethical Design** | Decision-support tool with clear warnings against autonomous diagnosis |

---

## 🏗️ How It Works — Step by Step

### Step 1 — Image Input
The user uploads a Chest X-ray via the web UI or CLI. The image is:
- Resized to **224×224 pixels**
- Normalized using ImageNet statistics (mean/std)
- Converted to a PyTorch tensor

### Step 2 — Forward Pass (MiniMedNet)
The image passes through **MiniMedNet**, a ResNet18-based classifier:
- 4 residual blocks with batch normalization and ReLU activation
- Global Average Pooling replaces fully-connected flattening
- A single output neuron outputs a **logit** (raw score)
- `sigmoid(logit)` converts it to a **probability [0, 1]**

### Step 3 — Diagnosis Decision
- Probability **≥ 0.5** → **Pneumonia** 🔴
- Probability **< 0.5** → **Normal** 🟢
- Confidence is categorized as **High** (>80% or <20%), **Moderate** (>70% or <30%), or **Low** (everything else)

### Step 4 — Grad-CAM Explanation
- Gradients of the class output w.r.t. the **last convolutional layer** (`layer4`) are computed
- Global Average Pooling on gradients gives **importance weights** per channel
- Weighted sum of feature maps → raw CAM → **ReLU** → upsampled to 224×224
- Result is a **float32 heatmap** normalized to [0, 1]

### Step 5 — Overlay Visualization
- Heatmap is colored using a **JET colormap** (blue=low, red=high impact)
- Blended with the original X-ray at configurable alpha transparency
- Returned as both **NumPy BGR array** and **PIL Image**

### Step 6 — Dashboard Display
The results are displayed in the browser with:
- Original X-ray and Grad-CAM overlay side by side
- Probability gauge, label badge, and confidence level
- Full scan history from the database

---

## 📁 Project Structure

```
mini-med-net/
│
├── app.py                          # 🚀 Flask web app entry point
│
├── src/                            # Core AI Engine
│   ├── architectures/
│   │   └── mini_med_net.py         # ResNet18-based CNN model definition
│   ├── explainability/
│   │   ├── gradcam.py              # Grad-CAM implementation
│   │   └── overlay.py              # Heatmap colorization & overlay
│   ├── inference/
│   │   └── predictor.py            # End-to-end MiniMedPredictor class
│   ├── training/
│   │   ├── dataset.py              # ChestXRayDataset with augmentation
│   │   ├── train.py                # Training loop
│   │   └── evaluate.py             # Metrics, plots, confusion matrix
│   └── utils/
│       └── image_utils.py          # Image loading/preprocessing
│
├── scripts/
│   ├── predict.py                  # CLI prediction script
│   ├── generate_demo_model.py      # Generate random demo weights
│   ├── run_training.bat            # Windows training launcher
│   └── run_training.sh             # Linux/macOS training launcher
│
├── tests/
│   ├── test_model.py               # Model architecture tests
│   ├── test_gradcam.py             # Grad-CAM unit tests
│   └── test_predictor.py           # End-to-end predictor tests
│
├── templates/
│   └── index.html                  # Flask HTML dashboard
│
├── static/                         # CSS, JS, uploaded images
├── config/
│   └── model_config.yaml           # Model hyperparameters
├── models/                         # Saved model weights (.pth)
├── data/                           # Datasets & scan history DB
├── docs/
│   ├── system_overview.md
│   └── methodology.md
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Pytest + project config
└── setup.py                        # Package setup for src imports
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python **3.10+**
- pip
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Divyapatel19/mini-med-net.git
cd mini-med-net
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Demo Model Weights
If you don't have trained weights, generate dummy weights for testing:
```bash
python scripts/generate_demo_model.py
```
*This creates `models/mini_med_net_demo.pth` with random weights.*

---

## 🌐 Running the Web Dashboard

```bash
python app.py
```

Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

You will see the **Transparent Mini-Med Clinical Dashboard** where you can:
1. Upload a chest X-ray image (JPG/PNG)
2. View the **diagnosis result** (Pneumonia / Normal)
3. View the **confidence percentage**
4. See the **Grad-CAM heatmap overlay**
5. Browse your **scan history**

---

## 💻 CLI Usage

### Predict a Single Image
```bash
python scripts/predict.py --image data/samples/sample_xray.jpeg
```

### Predict with custom threshold
```bash
python scripts/predict.py --image path/to/xray.jpg --threshold 0.6
```

---

## 🏋️ Training Your Own Model

### Step 1 — Download the Dataset

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
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Step 2 — Train the Model
```bash
# Windows
scripts\run_training.bat

# Linux / macOS
bash scripts/run_training.sh
```

Or run directly:
```bash
python src/training/train.py --config config/model_config.yaml
```

### Step 3 — Evaluate the Model
```bash
python src/training/evaluate.py \
  --weights models/mini_med_net_best.pth \
  --data-dir data/chest_xrays/test
```

This outputs:
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrix (saved to `outputs/confusion_matrix.png`)
- ROC Curve (saved to `outputs/roc_curve.png`)

---

## 🧪 Running Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src
```

| Test File | What It Tests |
|---|---|
| `test_model.py` | Model output shape, parameter count, Grad-CAM layer, gradient flow |
| `test_gradcam.py` | Heatmap shape, range, dtype, hook cleanup, overlay utilities |
| `test_predictor.py` | Full pipeline, label validity, confidence labels, PIL input |

---

## 🔬 Model Architecture

```
Input: (B, 3, 224, 224)     ← RGB chest X-ray image
     │
     ▼
ResNet18 Backbone
  - Conv1 + BN + ReLU + MaxPool
  - Layer1: 2× BasicBlock (64 channels)
  - Layer2: 2× BasicBlock (128 channels)
  - Layer3: 2× BasicBlock (256 channels)
  - Layer4: 2× BasicBlock (512 channels)  ← Grad-CAM target
     │
     ▼
Global Average Pooling  → (B, 512)
     │
     ▼
Linear(512 → 1)          ← Binary classifier
     │
     ▼
Output: (B, 1) logit     → sigmoid → Probability ∈ [0, 1]
```

**Total Parameters:** ~11.2 Million  
**Training Loss:** Binary Cross Entropy with Logits  
**Optimizer:** Adam  

---

## 🔥 Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) explains the model's prediction:

1. Forward pass → compute class probability
2. Backpropagate gradients to `layer4` (last convolutional block)
3. Global Average Pool the gradients → importance weight per feature map channel
4. Weighted sum of feature maps → raw class activation map
5. Apply ReLU → remove negative (irrelevant) activations
6. Upsample to 224×224 → normalize to [0, 1]

**Color Interpretation:**
- 🔴 **Red/Yellow** — high influence regions (where model "looks")
- 🔵 **Blue** — low influence regions

---

## 🛡️ Ethical Statement

This project is a **clinical decision-support research tool**, not an autonomous diagnostic engine:

- Physicians must review all predictions before any clinical decisions
- The model is not validated across diverse hospital equipment or patient demographics
- FDA/EMA regulatory approval is required before any clinical deployment
- Misuse for autonomous diagnosis without clinical oversight is strongly discouraged
- Training data bias may affect results across different demographic groups

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `flask` | Web server for dashboard |
| `opencv-python` | Image loading and heatmap colorization |
| `Pillow` | PIL image handling |
| `numpy` | Array operations |
| `matplotlib`, `seaborn` | Plotting evaluation charts |
| `scikit-learn` | Metrics (ROC-AUC, F1, etc.) |
| `tqdm` | Training progress bars |
| `pytest` | Unit testing framework |
| `PyYAML` | Config file parsing |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments

- **Kaggle** — Chest X-Ray Images (Pneumonia) dataset by Paul Mooney
- **PyTorch** — Deep learning framework
- **Grad-CAM** — Selvaraju et al., 2017 ([Paper](https://arxiv.org/abs/1610.02391))
- **ResNet** — He et al., 2016 ([Paper](https://arxiv.org/abs/1512.03385))

---

<p align="center">
  <i>Transparent Mini-Med — Explainable AI, Built for Trust. 🩺</i>
</p>
