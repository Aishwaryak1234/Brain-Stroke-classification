# Brain Stroke Classification using Hybrid ViT-CNN Model

## 🧠 Project Overview

This repository contains the implementation of a **Hybrid Vision Transformer (ViT) and Convolutional Neural Network (CNN)** model for classifying brain stroke types from CT scan images. The model distinguishes between **ischemic** and **hemorrhagic** strokes using advanced deep learning techniques.

**Project Type**: MSc Artificial Intelligence Thesis Project  
**Domain**: Medical Image Analysis / Computer Vision  
**Classification Task**: Binary Classification (Ischemic vs Hemorrhagic Stroke)

## 🎯 Key Features

- **Hybrid Architecture**: Combines ResNet18 CNN with ViT-Base for complementary feature extraction
- **Advanced Preprocessing**: CLAHE enhancement, skull isolation, and noise removal
- **K-Fold Cross-Validation**: Robust evaluation with 3-fold stratified cross-validation
- **Data Augmentation**: MixUp, geometric transformations, and color jittering
- **Explainability**: LIME analysis for model interpretability
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, ROC curves, and confusion matrices

## 🏗️ Architecture

### Model Components

1. **CNN Branch (ResNet18)**
   - Pre-trained on ImageNet
   - Extracts local spatial features and textures
   - Outputs 512-dimensional feature vector

2. **ViT Branch (Vision Transformer Base)**
   - Pre-trained on ImageNet
   - Captures global spatial relationships and attention patterns
   - Outputs 768-dimensional feature vector

3. **Feature Fusion**
   - Concatenates CNN and ViT features (1280 total features)
   - Dropout regularization (0.5)
   - Final classification layer (2 classes)

### Training Strategy

- **Selective Layer Freezing**: Freezes early layers, fine-tunes later layers
- **Differential Learning Rates**: Different rates for CNN, ViT, and classifier
- **Gradient Accumulation**: Effective batch size of 32 with accumulation steps
- **Early Stopping**: Prevents overfitting with patience-based stopping

## 📊 Performance Results

### Cross-Validation Performance (3-Fold)
- **Validation Accuracy**: 96.62% ± 1.58%
- **Validation F1-Score**: 96.61% ± 1.60%
- **Validation Precision**: 96.86% ± 1.37%
- **Validation Recall**: 96.62% ± 1.58%

### Test Set Performance (Best Model)
- **Test Accuracy**: 96.88%
- **Test F1-Score**: 96.87%
- **Ischemic Stroke**: 94% precision, 100% recall
- **Hemorrhagic Stroke**: 100% precision, 93% recall

## 🛠️ Technical Requirements

### Dependencies
```bash
pip install torch torchvision transformers torchcam scikit-learn pillow numpy opencv-python seaborn matplotlib
```

### Key Libraries
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face ViT implementation
- **OpenCV**: Image processing and computer vision
- **scikit-learn**: Machine learning utilities and metrics
- **LIME**: Model explainability framework

### Hardware Requirements
- **GPU**: Recommended (CUDA-compatible)
- **Memory**: Minimum 8GB RAM
- **Storage**: 2GB+ for models and datasets

## 📁 Project Structure

```
final-vit-cnn.ipynb          # Main implementation notebook
├── Section 1: Environment Setup
├── Section 2: Image Preprocessing Pipeline
├── Section 3: Data Preprocessing & Visualization
├── Section 4: Dataset Class & Data Augmentation
├── Section 5: Data Loading & Validation
├── Section 6: Evaluation & Plotting Functions
├── Section 7: Cross-Validation Setup
├── Section 8: Model Architecture Definition
├── Section 9: K-Fold Cross-Validation Training
├── Section 10: Results Analysis
├── Section 11: Advanced Visualizations
├── Section 12: Model Selection & Test Evaluation
├── Section 13: LIME Explainability Analysis
└── Section 14: Hyperparameter Optimization
```

## 🔧 Data Preprocessing Pipeline

### 1. Image Cleaning
- Background noise removal using thresholding
- Skull region isolation via connected components analysis
- Border artifact removal

### 2. Image Enhancement
- **Cropping**: 512×512 → 400×400 → 224×224
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Normalization**: Intensity scaling to [0, 255] range

### 3. Data Splitting
- **Training**: 80% of data
- **Validation**: 10% of data  
- **Test**: 10% of data (held out for final evaluation)

## 🚀 Usage Instructions

### 1. Environment Setup
```python
# Install required packages
!pip install torch torchvision transformers torchcam scikit-learn pillow numpy opencv-python -q
```

### 2. Data Preparation
```python
# Execute preprocessing pipeline
preprocess_and_split('/path/to/input/dataset', '/path/to/output/dataset')
```

### 3. Model Training
```python
# Initialize hybrid model
model = HybridViTCNN(num_classes=2)

# Setup training configuration
EPOCHS = 50
BATCH_SIZE = 8
PATIENCE = 8
```

### 4. Evaluation
```python
# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
    # Training and validation loop
    # Model checkpointing and early stopping
```

## 📈 Key Results & Visualizations

### Training Curves
- Loss vs. Epoch curves for training and validation
- Accuracy vs. Epoch curves with smoothing
- Learning rate scheduling visualization

### Performance Metrics
- Confusion matrices (raw counts and normalized)
- ROC curves and Precision-Recall curves
- Per-fold performance comparison

### Explainability
- LIME explanations for both stroke types
- Brain region highlighting for clinical interpretation
- Feature importance visualization

## 🔬 Model Interpretability

### LIME Analysis
- **Local Interpretable Model-agnostic Explanations**
- Highlights brain regions contributing to classification decisions
- Clinical relevance through brain tissue masking
- Comparative analysis between ischemic and hemorrhagic cases

### Visualization Features
- Original image with predictions
- LIME explanation boundaries
- Heatmap overlays for clinical interpretation
- Brain-masked explanations focusing on relevant regions

## 📚 Research Contributions

### Technical Innovations
1. **Hybrid Architecture**: Novel combination of CNN and ViT for medical imaging
2. **Advanced Preprocessing**: Comprehensive pipeline for CT scan enhancement
3. **Robust Evaluation**: K-fold cross-validation with comprehensive metrics
4. **Explainability**: Clinical interpretation through LIME analysis

### Medical Relevance
- **Stroke Classification**: Critical for treatment decision-making
- **Clinical Interpretability**: Understandable model decisions for healthcare professionals
- **Robust Performance**: Reliable classification across different CT scan variations

## 🎓 Academic Context

This project was developed as part of an **MSc in Artificial Intelligence** thesis, demonstrating:
- Advanced deep learning implementation
- Medical image analysis expertise
- Comprehensive evaluation methodology
- Research rigor and reproducibility

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{brain_stroke_classification_2024,
  title={Brain Stroke Classification using Hybrid ViT-CNN Model},
  author={[Your Name]},
  year={2024},
  note={MSc Artificial Intelligence Thesis Project}
}
```

## 🤝 Contributing

This is an academic research project. For questions or collaboration opportunities, please contact the author.

## 📄 License

This project is for academic research purposes. Please ensure compliance with relevant data usage and privacy regulations when working with medical imaging data.

## ⚠️ Important Notes

- **Medical Data**: This implementation is for research purposes only
- **Clinical Use**: Not validated for clinical decision-making
- **Data Privacy**: Ensure compliance with medical data regulations
- **Reproducibility**: Results may vary based on hardware and data variations

## 🔍 Future Work

Potential areas for improvement and extension:
- **Multi-class Classification**: Include other stroke types and normal cases
- **3D Analysis**: Extend to volumetric CT data
- **Real-time Processing**: Optimize for clinical deployment
- **Ensemble Methods**: Combine multiple model architectures
- **Transfer Learning**: Explore other pre-trained models

---

**Disclaimer**: This code is provided as-is for educational and research purposes. The authors are not responsible for any clinical decisions made based on this implementation.
