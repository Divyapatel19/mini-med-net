# Methodology — Transparent Mini-Med

## 1. Problem Statement

Traditional deep learning models for medical image analysis are accurate but opaque — clinicians cannot understand *why* a model made a particular prediction. This "black box" problem leads to distrust and limits clinical adoption. Transparent Mini-Med addresses this by combining a lightweight CNN with built-in visual explainability.

---

## 2. Dataset

**Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Origin:** Guangzhou Women and Children's Medical Center, published by Kermany et al. (Cell, 2018).

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

> Note: The validation set is very small. Training uses the WeightedRandomSampler to handle the class imbalance (~3:1 Pneumonia:Normal ratio).

---

## 3. Preprocessing Pipeline

1. **Resize** to 256×256
2. **Center Crop** to 224×224 (CNN input size)
3. **Training augmentation:**
   - RandomHorizontalFlip (p=0.5)
   - RandomRotation(±10°)
   - ColorJitter (brightness ±20%, contrast ±20%)
4. **Normalise** using ImageNet statistics: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

Validation and test sets skip augmentation and only apply resize + crop + normalise.

---

## 4. Model Architecture — ResNet18

Transparent Mini-Med utilizes a **ResNet18** backbone, a powerful deep residual network with ~11.2 million parameters.

```
Input (3×224×224)
│
├── Conv1 + BN + ReLU + MaxPool             → 64×112×112
├── Layer 1 (2 Residual Blocks)             → 64×56×56
├── Layer 2 (2 Residual Blocks)             → 128×28×28
├── Layer 3 (2 Residual Blocks)             → 256×14×14
├── Layer 4 (2 Residual Blocks)             → 512×7×7  (Grad-CAM Target)
│
├── Global Average Pool                     → 512
├── Dropout(0.5)                            → 512
└── FC(512→1) → Sigmoid                     → P(Pneumonia)
```

**Design rationale:** ResNet (Residual Network) skip-connections allow the model to learn much deeper and more complex features than a traditional CNN. This is crucial for medical imaging where pneumonia patterns can be subtle and easily confused with other structures.

---

## 5. Training Strategy

| Hyperparameter | Value |
|----------------|-------|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 30 (+ early stopping) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping patience | 10 |
| Best model metric | Val F1 |

**Class weighting:** The Normal class is given a slightly higher weight (1.5×) to compensate for the training imbalance.

---

## 6. Explainability — Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** (Selvaraju et al., 2017):

1. Forward pass → compute class score (logit)
2. Backward pass → compute gradient of score w.r.t. the last residual block (**layer4**)
3. **Importance weights** α_k = (1/Z) ΣΣ ∂y^c / ∂A^k_{ij}
4. **Saliency map** L^c = ReLU(Σ_k α_k · A_k)
5. **Resize** L^c to 224×224
6. **Normalise** to [0, 1], apply jet colormap, blend with original at opacity 0.4

**Why Layer 4?** The final residual blocks in ResNet18 capture the most semantically dense features. Since this is the highest abstraction level before the classifier, it highlights the "objects" (like opacity clusters or lung consolidation) that led to the final verdict.

---

## 7. Evaluation Metrics

| Metric | Formula | Clinical Relevance |
|--------|---------|-------------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | How often positive prediction is correct |
| Recall (Sensitivity) | TP/(TP+FN) | Critical — catching true pneumonia cases |
| F1 Score | 2·(P·R)/(P+R) | Balanced trade-off (used for checkpoint selection) |
| ROC-AUC | Area under ROC curve | Discrimination across all thresholds |

**Priority:** In medical diagnostics, **Recall (sensitivity)** is typically prioritised over Precision to minimise missed diagnoses (false negatives).

---

## 8. Limitations

- **Small validation set:** The 16-image val set is statistically insufficient for reliable early stopping — cross-validation would be more robust.
- **Single dataset bias:** Trained on one institution's imaging protocol; may not generalise to different scanners or populations.
- **Binary classification only:** Does not distinguish between bacterial/viral pneumonia or other pathologies.
- **No uncertainty estimation:** Confidence tiers are heuristic (distance from threshold); formal uncertainty metrics (MC-Dropout, Deep Ensembles) are not implemented.
- **Grad-CAM resolution:** Heatmaps are upsampled from the conv4 feature map resolution (28×28 before GAP); very fine-grained localisations may be imprecise.

---

## 9. References

1. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
2. Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122–1131.
3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
