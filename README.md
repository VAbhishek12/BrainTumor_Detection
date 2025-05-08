# 🧠 Brain Tumor Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to detect brain tumors from MRI scans. The goal is to assist in early detection and diagnosis using automated deep learning models trained on medical imaging data.

![Banner](https://github.com/user-attachments/assets/51b7be5e-d170-4e55-8f1d-638a689f1ef8)
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

![ModelArchitectureSummary](https://github.com/user-attachments/assets/9e4aca7f-4246-4baf-96f5-f3f3a511c244)

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
![AccuracyVsValidation](https://github.com/user-attachments/assets/ce7a6bd7-f66d-4f97-9a0d-02ac21f69e48)

#### ❌ Loss Curve
![LossVsValidation](https://github.com/user-attachments/assets/f56e732e-d986-4676-a821-a83571d5a99b)


---

### 🔍 Training Curve

![TrainingVSValidation](https://github.com/user-attachments/assets/569ebe32-e99f-4e94-83b3-e35a2fdd9d52)

---

### 🧪 Sample Predictions

| MRI Image | Predicted Label | Ground Truth |
|-----------|------------------|---------------|
|![Yes](https://github.com/user-attachments/assets/efd3914e-1eff-45f2-8e5a-d777885bf935)| Tumor        | Tumor        |
|![No](https://github.com/user-attachments/assets/b91eaf22-337d-47ea-a2ad-4b8e8672e31f)| No Tumor      | No Tumor     |

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
