# 🚀 Quick Start Guide - Jupyter Notebook Training

## Prerequisites
Ensure you have all dependencies installed:
```bash
pip install torch torchvision timm tqdm matplotlib seaborn scikit-learn jupyter notebook
```

## Step 1: Launch Jupyter Notebook
```bash
cd "c:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION"
jupyter notebook
```

## Step 2: Open Training Notebook
- In the browser, click on `train_notebook.ipynb`

## Step 3: Run Training
Execute cells in order:

1. **Cell 1**: Import libraries and check GPU
2. **Cell 2**: Configure settings (customize here!)
3. **Cell 3**: Load dataset
4. **Cell 4**: Visualize sample images
5. **Cell 5**: Create model
6. **Cell 6**: **START TRAINING** (this takes 10-30 minutes)
7. **Cells 7-8**: View results and plots

## Key Settings to Adjust (Cell 2)

### If you have a powerful GPU (8GB+ VRAM):
```python
config['batch_size'] = 64
config['num_workers'] = 4
config['use_amp'] = True
config['model_name'] = 'vit_small_patch16_224'  # or vit_base_patch16_224
```

### If you have limited GPU memory (4-6GB VRAM):
```python
config['batch_size'] = 32
config['num_workers'] = 2
config['use_amp'] = True  # Keep this on!
config['model_name'] = 'vit_tiny_patch16_224'
```

### For CPU training (slow, not recommended):
```python
config['batch_size'] = 16
config['num_workers'] = 0
config['use_amp'] = False
config['epochs'] = 10  # Reduce epochs
```

## Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total (30 epochs) |
|----------|-----------|----------------|-------------------|
| RTX 3090 | 64 | 1 min | ~30 min |
| RTX 3070 | 64 | 1.5 min | ~45 min |
| RTX 2060 | 32 | 2.5 min | ~75 min |
| GTX 1060 | 16 | 5 min | ~2.5 hours |
| CPU | 16 | 30+ min | ~15+ hours |

## Monitoring Progress

### During Training:
- Progress bars show real-time loss and accuracy
- Each epoch prints detailed metrics

### After Training:
- Plots automatically appear in notebook
- All files saved to `outputs/` folder

## What Gets Saved

```
outputs/
├── best_model.pth                    # Best model checkpoint
├── class_names.json                  # Class mappings
├── training_metrics.json             # All metrics (JSON)
├── accuracy_graph.png               # Accuracy plot
├── loss_graph.png                   # Loss plot
├── metrics_graph.png                # Precision/Recall/F1
├── confusion_matrix.png             # Confusion matrix
└── confusion_matrix_normalized.png  # Normalized CM
```

## Common Issues & Solutions

### ❌ "CUDA out of memory"
**Solution**: Reduce batch size
```python
config['batch_size'] = 32  # or 16
```

### ❌ "DataLoader worker process died"
**Solution**: Reduce workers
```python
config['num_workers'] = 0
```

### ❌ Training is very slow
**Check**:
1. Is GPU being used? (Should see "CUDA Available: True")
2. Is AMP enabled? (Should be `use_amp=True`)
3. Are workers running? (Should be `num_workers=4`)

### ❌ Accuracy not improving
**Try**:
1. More epochs: `config['epochs'] = 50`
2. Lower learning rate: `config['learning_rate'] = 1e-4`
3. Less augmentation (edit dataset_loader.py)

## Tips for Best Results

1. **Start with defaults** - They're already optimized!
2. **Monitor GPU usage** - Run `nvidia-smi` in terminal
3. **Use mixed precision** - Keep `use_amp=True`
4. **Increase batch size** - Use largest that fits in GPU
5. **Check early stopping** - Training stops if no improvement
6. **Save regularly** - Best model auto-saved each epoch

## Alternative: Run as Python Script

If you prefer command line:
```bash
python train.py
```

Settings are in `config.py`.

## Need Help?

1. Check [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for detailed explanations
2. Review error messages carefully
3. Try with `num_workers=0` for easier debugging

---

**Ready to train? Open `train_notebook.ipynb` and start with Cell 1! 🚀**
