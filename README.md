# 🧠 Brain Tumor Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to detect brain tumors from MRI scans. The goal is to assist in early detection and diagnosis using automated deep learning models trained on medical imaging data.

![Banner](assets/banner.png)

---

## 🚀 Project Overview

Brain tumors, if not diagnosed early, can pose serious health risks. This project leverages the power of CNNs to distinguish between tumorous and non-tumorous brain MRI images with high accuracy.

---

## 📁 Dataset

- **Source:** [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets)
- **Classes:** Tumor / No Tumor
- **Image Format:** JPEG/PNG
- **Preprocessing:**
  - Resizing to 224x224
  - Image normalization
  - Data augmentation (rotation, zoom, flips)

---

## 🧠 Model Architecture

- Input Layer: 224x224x3 images
- Conv2D + ReLU + MaxPooling
- Conv2D + ReLU + MaxPooling
- Flatten
- Dense Layer + ReLU
- Dropout (to prevent overfitting)
- Output Layer: Softmax for 2 classes (Tumor, No Tumor)

![Model Architecture](assets/model_architecture.png)

---

## 📊 Performance Metrics

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | 95%+      |
| Precision   | 94.8%     |
| Recall      | 95.5%     |
| F1-Score    | 95.1%     |

---

### 📉 Training Results

#### ✅ Accuracy Curve
![Accuracy](assets/accuracy_curve.png)

#### ❌ Loss Curve
![Loss](assets/loss_curve.png)

---

### 🔍 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

### 🧪 Sample Predictions

| MRI Image | Predicted Label | Ground Truth |
|-----------|------------------|---------------|
| ![img1](assets/sample1.png) | Tumor         | Tumor        |
| ![img2](assets/sample2.png) | No Tumor      | No Tumor     |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas / Matplotlib / Seaborn
- Jupyter Notebook
- OpenCV
- Google Colab / Local GPU

---

## 🧰 How to Run

1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection

2. Install dependencies
 ```
   pip install -r requirements.txt
```
3. Run the notebook
```
jupyter notebook BrainTumourDetection.ipynb
```
4. Or load the .h5 model and predict on new data
```
from tensorflow.keras.models import load_model
model = load_model('brain_tumor_model.h5')
```

# Future Work
 Improve accuracy using transfer learning (e.g., VGG16, ResNet)
 
 Add support for 3D volumetric scans (CT/MRI slices)
