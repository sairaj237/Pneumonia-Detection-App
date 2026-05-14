# Pneumonia Detection using ResNet18 (3-Class Chest X-Ray Classification)

## Overview

This project trains a deep learning model to classify chest X-ray images into three categories:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

The notebook uses transfer learning with a pretrained ResNet18 model from PyTorch.  
It includes:

- Custom dataset loader
- Data preprocessing
- Model training
- Validation/testing
- Confusion matrix visualization
- Model checkpoint saving

The goal is to build a medical image classification pipeline using convolutional neural networks (CNNs).

---

# Project Structure

```bash
project/
│
├── notebook1.ipynb
├── README.md
├── pneumonia_resnet18_3class.pt
│
└── chest_xray/
    ├── train/
    │   ├── normal/
    │   ├── pneumonia_bacteria/
    │   └── pneumonia_virus/
    │
    ├── val/
    │   ├── normal/
    │   ├── pneumonia_bacteria/
    │   └── pneumonia_virus/
    │
    └── test/
        ├── normal/
        ├── pneumonia_bacteria/
        └── pneumonia_virus/
```

---

# Features

- Transfer learning using ResNet18
- Weighted loss handling for class imbalance
- GPU support (CUDA)
- Confusion matrix visualization
- Easy-to-extend architecture
- Multi-class classification support

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Pretrained models and transforms |
| Scikit-learn | Evaluation metrics |
| PIL | Image loading |
| Matplotlib | Visualization |
| Seaborn | Heatmaps/confusion matrix |

---

# Dataset

The dataset contains chest X-ray images divided into:

| Class | Description |
|---|---|
| Normal | Healthy lungs |
| Pneumonia (Bacterial) | Bacterial lung infection |
| Pneumonia (Viral) | Viral lung infection |

Dataset directory structure is expected as:

```bash
chest_xray/
    train/
    val/
    test/
```

Each folder must contain class-specific subfolders.

Example:

```bash
train/
    normal/
    pneumonia_bacteria/
    pneumonia_virus/
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/sairaj237/Pneumonia-Detection-App.git

cd /Pneumonia-Detection-App
```

---

## 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install scikit-learn
pip install seaborn
```

Or create a requirements.txt file:

```txt
torch
torchvision
torchaudio
scikit-learn
seaborn
matplotlib
pillow
numpy
```

Install using:

```bash
pip install -r requirements.txt
```

---

# Model Architecture

The project uses:

## ResNet18

- Pretrained on ImageNet
- Final fully connected layer replaced for 3-class output

Original output layer:

```python
model.fc = nn.Linear(model.fc.in_features, 1000)
```

Modified output layer:

```python
model.fc = nn.Linear(model.fc.in_features, 3)
```

---

# Data Preprocessing

Images are transformed using:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

Why this matters:

- Resizing ensures consistent input dimensions
- Tensor conversion enables GPU processing
- Normalization matches ImageNet pretrained statistics

---

# Custom Dataset Loader

The notebook defines a custom PyTorch Dataset class:

```python
class PneumoniaFilenameDataset(Dataset):
```

Responsibilities:

- Read image files
- Assign labels
- Apply transforms
- Return tensors and labels

---

# Class Labels

| Label | Meaning |
|---|---|
| 0 | Normal |
| 1 | Bacterial Pneumonia |
| 2 | Viral Pneumonia |

---

# Handling Class Imbalance

Medical datasets are often imbalanced.

This project calculates class weights:

```python
weights = 1.0 / counts
```

Then applies them to:

```python
nn.CrossEntropyLoss(weight=weights)
```

This prevents the model from favoring majority classes.

---

# Training

## Optimizer

```python
torch.optim.Adam
```

Learning rate:

```python
1e-4
```

---

## Training Loop

The notebook trains for:

```python
10 epochs
```

Training steps:

1. Forward pass
2. Compute loss
3. Backpropagation
4. Update weights

---

# Evaluation

The project evaluates the model using:

```python
classification_report
```

Metrics include:

- Precision
- Recall
- F1-score
- Accuracy

---

# Confusion Matrix

A confusion matrix is generated using:

```python
confusion_matrix()
```

Visualized with:

```python
seaborn.heatmap()
```

This helps identify:

- False positives
- False negatives
- Misclassified pneumonia types

---

# Saving the Model

The trained model is saved as:

```bash
pneumonia_resnet18_3class.pt
```

Using:

```python
torch.save(model.state_dict(), ...)
```

---

# Running the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```bash
notebook1.ipynb
```

Run cells sequentially.

---

# Expected Results

Typical outputs:

- Training loss per epoch
- Classification report
- Confusion matrix heatmap
- Saved trained model

---

# Future Improvements

Possible upgrades:

## Model Improvements

- ResNet50
- EfficientNet
- DenseNet121

## Training Improvements

- Data augmentation
- Early stopping
- Learning rate scheduling
- Mixed precision training

## Evaluation Improvements

- ROC-AUC
- Grad-CAM visualization
- Precision-recall curves

## Deployment

- Flask/FastAPI API
- Streamlit web app
- Docker deployment

---

# Example Inference Code

```python
from PIL import Image
import torch

image = Image.open("sample.jpg")

model.eval()

with torch.no_grad():
    output = model(image)
    pred = torch.argmax(output, dim=1)

print(pred)
```

---

# Limitations

This project is educational and experimental.

Important limitations:

- Not medically certified
- Dataset quality strongly affects accuracy
- Real-world hospital deployment requires:
  - Clinical validation
  - Regulatory approval
  - Bias testing
  - Explainability tools

Do not use this system for actual medical diagnosis.

---

# Recommended Improvements for Production

If turning this into a real application:

- Add DICOM support
- Add explainable AI (Grad-CAM)
- Use larger medical datasets
- Add monitoring/logging
- Add confidence scores
- Use ensemble models

---

# License

MIT License

---


