# Vision Transformer Image Classification - Quick Start Guide

## Step-by-Step Instructions

### 1️⃣ Install Dependencies

Open a terminal and run:

```bash
pip install -r requirements.txt
```

For GPU support (recommended), install PyTorch with CUDA:
```bash
# Visit https://pytorch.org/ for your specific CUDA version
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ Verify Dataset

Make sure your dataset is structured correctly:
```
split_dataset/
├── train/    (training images in class folders)
├── val/      (validation images in class folders)
└── test/     (test images in class folders)
```

### 3️⃣ Train the Model

```bash
python train.py
```

**What happens:**
- Loads train/val datasets
- Creates ViT model with pretrained weights
- Trains for 30 epochs
- Saves best model to `outputs/best_model.pth`
- Generates training curves

**Expected time:** 1-2 hours with GPU, 8-12 hours with CPU

### 4️⃣ Evaluate Performance

```bash
python evaluate.py
```

**What happens:**
- Loads the best trained model
- Evaluates on test dataset
- Generates confusion matrix
- Saves detailed metrics

**Output files:**
- `outputs/test_metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`

### 5️⃣ Make Predictions

#### Option A: Predict a single image
```bash
python inference.py --mode single --image "path/to/your/image.png"
```

#### Option B: Predict all images in a folder
```bash
python inference.py --mode directory --dir "path/to/images/" --output results.csv
```

#### Option C: Generate prediction table (recommended)
```bash
python inference.py --mode table
```

This generates a table with Image Name | True Label | Predicted Label

---

## ⚡ Quick Commands

### Test dataset loader (before training)
```bash
python dataset_loader.py
```

### Test model creation (before training)
```bash
python model.py
```

### Full pipeline
```bash
python train.py && python evaluate.py && python inference.py --mode table
```

---

## 🔧 Configuration

Edit `config.py` or directly in `train.py` to change:

- **Model size**: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`
- **Batch size**: Default 32 (reduce to 16 or 8 if GPU memory issues)
- **Epochs**: Default 30 (increase for better performance)
- **Learning rate**: Default 1e-4

---

## 📊 Expected Results

After training, you should see:
- Training accuracy: ~90-98%
- Validation accuracy: ~85-95%
- Test accuracy: ~85-95%

Results depend on your dataset quality and size.

---

## ❓ Common Issues

**Problem:** CUDA out of memory
**Solution:** Reduce batch size in `train.py`:
```python
'batch_size': 16,  # or 8
```

**Problem:** Slow data loading
**Solution:** Change `num_workers` in `train.py`:
```python
'num_workers': 0,  # on Windows
```

**Problem:** Model not found
**Solution:** Make sure you run `python train.py` first

---

## 📈 Monitor Training

Watch for these signs of good training:
- ✅ Training loss steadily decreasing
- ✅ Validation loss following training loss
- ✅ Accuracy increasing over epochs
- ⚠️ If validation loss increases: reduce learning rate

---

## 🎯 Next Steps

1. **Experiment with different models:**
   - Try `vit_small_patch16_224` for faster training
   - Try `vit_large_patch16_224` for better accuracy

2. **Fine-tune hyperparameters:**
   - Adjust learning rate
   - Increase epochs
   - Modify data augmentation

3. **Analyze results:**
   - Check confusion matrix for problematic classes
   - Review classification report for per-class performance
   - Examine misclassified samples

---

**Ready to start? Run:** `python train.py` 🚀
