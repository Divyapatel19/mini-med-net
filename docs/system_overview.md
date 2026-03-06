# System Overview: Transparent Mini-Med 🩺

**Medical Image Diagnostic AI System**  
Transparent Mini-Med is a state-of-the-art **Clinical Decision Support Tool** designed to assist radiologists and physicians in detecting pneumonia from chest X-ray images with high accuracy and transparency.

---

## 🏗️ Technical Foundation
- **Lightweight Disease Detection**: Optimized for speed and clinical efficiency.
- **CNN Model**: Utilizes a robust **ResNet18 Backbone** (~11.2M parameters) for deep feature extraction.
- **High Performance**: Trained to identify subtle opacities and lung consolidation patterns.

---

## 🧠 Explainability Layer
> **આ projectનું heart છે.** (This is the heart of the project.)

The system bridges the gap between AI accuracy and **Doctor Trust** by providing clear visual evidence for every diagnostic verdict.

### Grad-CAM Technique
*Gradient-weighted Class Activation Mapping*

👉 **Modelના last convolution layer gradients use કરે છે**  
(Uses gradients from the last residual block/convolution layer to identify importance.)

👉 **Important image regions highlight કરે છે.**  
(Directly highlights the regions of the lung that led to the diagnosis.)

### Heatmap Analysis:
🔥 **Red** = High importance (AI is focused here)  
❄️ **Blue** = Low importance (AI ignores these regions)

### Technical Logic:
`Gradients` → `Feature maps` → `Weighted activation` → `Heatmap`

### Clinical Value:
**Doctor જોઈ શકે:**  
👉 **Disease ક્યાં છે.** (The exact localization of the pathology for surgical or treatment planning.)

---

## 🖥️ Output Layer (Clinical Interface)
The interface is designed for maximum clarity, providing the doctor with four critical data points for every scan:

1. **Original X-ray**: The raw medical image for comparison.
2. **AI Prediction**: Clear verdict (Normal / Pneumonia).
3. **Heatmap Overlay**: Visual proof of the "Explainability Layer."
4. **Confidence Score**: Statistical probability of the diagnosis.

### ✅ Example Output:
- **Diagnosis**: Pneumonia
- **Probability**: 0.89 (89%)
- **Affected Region**: Right lower lung (Identified via Heatmap)

---

> ⚠️ **Clinical Advisory**: This system is meant to support, not replace, medical experts. All AI results should be verified by a board-certified radiologist.
