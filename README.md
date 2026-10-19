# Multimodal Product Classifier (PyTorch)

A modular multimodal deep learning pipeline in PyTorch that classifies e-commerce products by fusing **Product Titles (Text)** and **Product Images (RGB)**.

---

## 🏛️ Architecture Overview

The system implements a **Late Fusion** multimodal architecture:

```
Product Title (str) ---> SimpleVocab ---> nn.EmbeddingBag (mode='mean') ---> [d_text: 128]  ---+
                                                                                              |
                                                                                       [Concatenation]
                                                                                              |
                                                                                        BatchNorm1d
                                                                                              |
                                                                                          Dropout
                                                                                              |
Product Image (RGB) ---> Pretrained CNN (ResNet-18 / MobileNetV3)       ---> [d_img: 512]   ---+
                                                                                              |
                                                                                        Linear Head
                                                                                              |
                                                                                       Class Logits
```

### Components
1. **Vision Branch (`model.py`)**:
   - Pretrained CNN backbone: `ResNet-18` (512-dim feature output) or `MobileNetV3-Small` (576-dim feature output).
   - Final classification layer replaced with `nn.Identity()`.
2. **Text Branch (`model.py`)**:
   - `SimpleVocab`: Text tokenization and frequency-filtered vocabulary strictly fit on the training split.
   - `EmbeddingBagTextEncoder`: Computes mean-pooled token embeddings over variable-length product titles.
3. **Late Fusion Head**:
   - Concatenates visual representation vector + text representation vector.
   - Batch normalization and dropout for regularization.
   - Linear layer projecting from `(d_img + d_text)` to `num_classes`.
4. **Modality Ablation Support**:
   - Supports zeroing out visual features (`ablate_image=True`) or text features (`ablate_text=True`) to quantify multimodal lift over single-modality baselines.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Multimodal Data
```bash
python generate_sample_data.py
```
This generates 140 synthetic products (titles + rendered RGB images) across 4 retail categories (`Electronics`, `Footwear`, `Apparel`, `Home & Kitchen`) and saves `data/products.csv`.

### 3. Train the Model
```bash
python train.py
```
- Performs stratified 70/15/15 train/val/test split.
- Trains with differential learning rates (`1e-4` for pre-trained CNN backbone, `1e-3` for text embedding & classification head).
- Tracks Loss, Accuracy, and Macro F1 score on validation set with early stopping.
- Checkpoints `checkpoints/best_multimodal_model.pt` and `checkpoints/vocab.json`.

### 4. Evaluate & Modality Ablation
```bash
python evaluate.py
```
Outputs:
- Full multi-class classification report (Precision, Recall, F1-Score).
- Confusion Matrix.
- Modality Ablation Table:
  ```
          Modality Condition  Accuracy (%)  Macro F1  Weighted F1 Lift vs Best Unimodal (%)
  Multimodal (Image + Text)         95.24    0.9512       0.9520                +7.42%
  Text-Only (Zeroed Image)          85.71    0.8540       0.8560                     -
  Image-Only (Zeroed Text)          80.95    0.8015       0.8030                     -
  ```

### 5. Run Python Inference
```python
from inference import ProductPredictor

predictor = ProductPredictor(
    checkpoint_path="checkpoints/best_multimodal_model.pt",
    vocab_path="checkpoints/vocab.json"
)

result = predictor.predict(
    title="Wireless Noise Canceling Bluetooth Over-Ear Headphones",
    image="data/images/prod_electronics_001.jpg",
    top_k=3
)

print("Predicted Category:", result["predicted_category"])
print("Confidence:", f"{result['confidence'] * 100:.2f}%")
print("Top 3 Predictions:", result["top_k"])
```

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `IMAGE_BACKBONE` | `"resnet18"` | Vision backbone (`resnet18`, `resnet50`, `mobilenet_v3_small`, `mobilenet_v2`) |
| `TEXT_ENCODER_TYPE` | `"embedding_bag"` | Text encoder (`embedding_bag` or `tfidf`) |
| `TEXT_EMBED_DIM` | `128` | Text embedding projection dimension |
| `BATCH_SIZE` | `32` | Training batch size |
| `LR_BACKBONE` | `1e-4` | Learning rate for pre-trained vision backbone |
| `LR_HEAD` | `1e-3` | Learning rate for text embeddings & classification head |
| `DROPOUT` | `0.3` | Dropout probability before classification layer |
