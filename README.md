# Vision Transformer Image Classification Project

A complete image classification system using Vision Transformer (ViT) with transfer learning for multi-class classification of grayscale speckle-pattern images.

## 📋 Project Overview

This project implements a state-of-the-art image classification pipeline using Vision Transformer (ViT) architecture. The system is designed for classifying grayscale images organized in class-wise folders, with automatic class inference and comprehensive evaluation metrics.

### Key Features

- ✅ **Vision Transformer (ViT)**: Utilizes pretrained ViT models via timm library
- ✅ **Transfer Learning**: Leverages pretrained weights for improved performance
- ✅ **Automatic Preprocessing**: Converts grayscale images to RGB format
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- ✅ **Easy Inference**: Single-image and batch prediction capabilities
- ✅ **Well-Structured Code**: Modular design with clear separation of concerns

## 📁 Project Structure

```
IMAGE_CLASSIFICATION/
├── dataset_loader.py       # Dataset loading and preprocessing
├── model.py                # Vision Transformer model definition
├── train.py                # Training loop and model training
├── evaluate.py             # Model evaluation and metrics
├── inference.py            # Single-image and batch inference
├── config.py               # Configuration parameters
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── split_dataset/         # Pre-split dataset
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   ├── test/              # Test images
│   └── unused/            # (Excluded from training)
│
└── outputs/               # Generated during training
    ├── best_model.pth     # Best model checkpoint
    ├── final_model.pth    # Final model checkpoint
    ├── class_names.json   # Class names mapping
    ├── training_curves.png
    ├── confusion_matrix.png
    └── predictions.csv
```

## 🚀 Quick Start

### 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

**Important**: For GPU support, install PyTorch with CUDA support:
- Visit https://pytorch.org/ for installation instructions specific to your system
- Example for CUDA 11.8: 
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 2. Verify Dataset Structure

Ensure your dataset is organized as follows:

```
split_dataset/
├── train/
│   ├── l=-32/
│   ├── l=-31/
│   └── ... (all class folders)
├── val/
│   ├── l=-32/
│   └── ...
└── test/
    ├── l=-32/
    └── ...
```

### 3. Train the Model

Run the training script:

```bash
python train.py
```

This will:
- Load the training and validation datasets
- Create a pretrained ViT model
- Train for 30 epochs (configurable in `config.py`)
- Save the best model based on validation accuracy
- Generate training curves

**Training Output:**
- `outputs/best_model.pth` - Best model checkpoint
- `outputs/final_model.pth` - Final model checkpoint
- `outputs/training_curves.png` - Loss and accuracy plots
- `outputs/class_names.json` - Class names mapping

### 4. Evaluate the Model

After training, evaluate on the test set:

```bash
python evaluate.py
```

This will:
- Load the best trained model
- Evaluate on the test dataset
- Generate comprehensive metrics and visualizations

**Evaluation Output:**
- `outputs/test_metrics.json` - Accuracy, Precision, Recall, F1-score
- `outputs/confusion_matrix.png` - Confusion matrix visualization
- `outputs/confusion_matrix_normalized.png` - Normalized confusion matrix
- `outputs/classification_report.txt` - Detailed per-class metrics
- `outputs/predictions.csv` - All predictions with labels

### 5. Inference

#### a) Single Image Prediction

```bash
python inference.py --mode single --image path/to/image.png
```

#### b) Batch Prediction (Directory)

```bash
python inference.py --mode directory --dir path/to/images/ --output predictions.csv
```

#### c) Generate Prediction Table (with true labels from test set)

```bash
python inference.py --mode table --test-dir split_dataset/test --output outputs/predictions.csv
```

This generates a table showing:
- Image Name
- True Label
- Predicted Label
- Confidence
- Correctness (✓/✗)

## ⚙️ Configuration

Edit [`config.py`](config.py) to customize:

```python
# Model Selection
MODEL_CONFIG = {
    'model_name': 'vit_base_patch16_224',  # ViT variant
    'pretrained': True,
    'freeze_backbone': False,
}

# Training Parameters
TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 1e-4,
    'batch_size': 32,
}
```

### Available ViT Models

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|----------|
| `vit_tiny_patch16_224` | 5.7M | Fastest | Good |
| `vit_small_patch16_224` | 22M | Fast | Better |
| `vit_base_patch16_224` | 86M | Moderate | Best (Recommended) |
| `vit_large_patch16_224` | 304M | Slow | Highest |

## 📊 Evaluation Metrics

The evaluation script computes:

1. **Overall Metrics**:
   - Accuracy
   - Weighted Precision
   - Weighted Recall
   - Weighted F1-Score

2. **Per-Class Metrics**:
   - Precision per class
   - Recall per class
   - F1-score per class

3. **Visualizations**:
   - Confusion Matrix
   - Normalized Confusion Matrix
   - Training/Validation Curves

## 💡 Usage Examples

### Example 1: Quick Training and Evaluation

```bash
# Train the model
python train.py

# Evaluate on test set
python evaluate.py

# Generate predictions table
python inference.py --mode table
```

### Example 2: Custom Configuration

Modify `train.py` or `config.py` for custom settings:

```python
config = {
    'epochs': 50,
    'learning_rate': 5e-5,
    'batch_size': 16,
    'model_name': 'vit_small_patch16_224',
}
```

### Example 3: Predict Single Image in Python

```python
from inference import ImageClassifier
import json

# Load class names
with open('outputs/class_names.json', 'r') as f:
    class_names = json.load(f)

# Create classifier
classifier = ImageClassifier(
    model_path='outputs/best_model.pth',
    class_names=class_names
)

# Predict
result = classifier.predict('path/to/image.png')
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `config.py` or `train.py`:
```python
'batch_size': 16,  # or 8
```

### Issue: Slow Data Loading on Windows

**Solution**: Set `num_workers=0` in `config.py`:
```python
'num_workers': 0,
```

### Issue: Model Not Found

**Solution**: Ensure you've run training first:
```bash
python train.py
```

## 📈 Training Tips

1. **Start with a Smaller Model**: Use `vit_small_patch16_224` for faster experimentation
2. **Monitor Validation Loss**: If validation loss increases while training loss decreases, reduce learning rate
3. **Use Learning Rate Scheduler**: The default `plateau` scheduler automatically reduces LR when validation loss plateaus
4. **Data Augmentation**: Training transforms include augmentation (flip, rotation, color jitter)

## 🎯 Expected Performance

Performance depends on:
- Dataset size and quality
- Number of classes
- Model variant chosen
- Number of training epochs

Typical results with ViT-Base:
- Training time: ~1-2 hours (with GPU, 30 epochs)
- Validation accuracy: 85-95% (dataset dependent)
- Inference speed: ~50-100 images/second (with GPU)

## 📝 Notes

- **Grayscale Images**: Automatically converted to RGB (3-channel) to match ViT requirements
- **Image Size**: All images resized to 224×224 (ViT standard)
- **Normalization**: Uses ImageNet mean and std for pretrained models
- **Class Inference**: Classes automatically inferred from folder names
- **Unused Folder**: The "unused" folder is automatically excluded

## 🤝 Contributing

To modify or extend the project:

1. **Add Custom Transforms**: Edit `get_transforms()` in [`dataset_loader.py`](dataset_loader.py)
2. **Change Model Architecture**: Modify `create_model()` in [`model.py`](model.py)
3. **Customize Training**: Edit the `Trainer` class in [`train.py`](train.py)
4. **Add Metrics**: Extend `ModelEvaluator` in [`evaluate.py`](evaluate.py)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **timm**: PyTorch Image Models library by Ross Wightman
- **Vision Transformer**: "An Image is Worth 16x16 Words" by Dosovitskiy et al.

---

**Happy Training! 🚀**
