# 🚀 Quick Commands Reference

## Push to GitHub (3 Simple Steps)

### Step 1: Create GitHub Repository
1. Go to: https://github.com/new
2. Create a new repository (e.g., "image-classification")
3. Copy the repository URL (e.g., `https://github.com/username/image-classification.git`)

### Step 2: Run These Commands

Open PowerShell in your project directory and run:

```powershell
# Navigate to project
cd "c:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION"

# Stage all files
git add .

# Commit changes
git commit -m "Optimized training pipeline with AMP, OneCycleLR, and Jupyter support"

# Add remote (REPLACE with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git push -u origin main
```

### Step 3: Verify
Go to your GitHub repository URL to see your code!

---

## Clone & Use in Google Colab

### Quick Setup (Copy-Paste in Colab)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone your repository (REPLACE with your GitHub URL)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 3. Navigate to repo
import os
os.chdir('/content/YOUR_REPO_NAME')

# 4. Install dependencies
!pip install -q timm tqdm scikit-learn seaborn

# 5. Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("✅ Setup complete!")
```

---

## Mount Google Drive (Colab)

### Simple Mount
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Mount & Link Dataset
```python
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

# Link dataset (fast, no copying)
drive_data = '/content/drive/My Drive/Datasets/split_dataset'
local_link = '/content/split_dataset'

if os.path.exists(drive_data):
    !ln -s "{drive_data}" "{local_link}"
    print(f"✅ Dataset linked: {local_link}")
else:
    print(f"❌ Upload dataset to: {drive_data}")
```

---

## Complete Colab Training (One Cell)

```python
# ===== COMPLETE SETUP & TRAINING =====

# 1. Mount Drive & Clone
from google.colab import drive
import os
drive.mount('/content/drive')

REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"  # CHANGE THIS
REPO_NAME = "IMAGE_CLASSIFICATION"  # CHANGE THIS

if not os.path.exists(f'/content/{REPO_NAME}'):
    !git clone {REPO_URL}
os.chdir(f'/content/{REPO_NAME}')

# 2. Install dependencies
!pip install -q timm tqdm scikit-learn seaborn

# 3. Import & Configure
import torch
from dataset_loader import create_dataloaders
from model import create_model
from train import Trainer
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG

config = {**DATA_CONFIG, **MODEL_CONFIG, **TRAIN_CONFIG, **OUTPUT_CONFIG}
config['data_root'] = 'split_dataset'  # Adjust path
config['batch_size'] = 64
config['num_workers'] = 2
config['output_dir'] = '/content/drive/My Drive/outputs'  # Save to Drive

# 4. Load Data
train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
    config['data_root'], config['batch_size'], config['num_workers'], 
    config['image_size']
)

# 5. Create Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes, config['model_name'], True, False, 0.1)

# 6. Train
trainer = Trainer(model, train_loader, val_loader, device, config)
trainer.train()

print("✅ Training complete! Check Google Drive for results.")
```

---

## Copy Files to/from Google Drive

### Save Outputs to Drive
```python
import shutil
import os

# Copy outputs to Drive
output_dir = '/content/IMAGE_CLASSIFICATION/outputs'
drive_backup = '/content/drive/My Drive/training_outputs'

if os.path.exists(output_dir):
    os.makedirs(drive_backup, exist_ok=True)
    shutil.copytree(output_dir, drive_backup, dirs_exist_ok=True)
    print(f"✅ Saved to: {drive_backup}")
```

### Copy Dataset from Drive
```python
import shutil

# Copy dataset to Colab (faster training)
drive_data = '/content/drive/My Drive/Datasets/split_dataset'
local_data = '/content/split_dataset'

if not os.path.exists(local_data):
    print("📥 Copying dataset...")
    shutil.copytree(drive_data, local_data)
    print("✅ Dataset ready!")
```

---

## Git Workflow (After First Push)

### Update Existing Repository
```powershell
# Make changes to your code...

# Check what changed
git status

# Stage changes
git add .

# Commit
git commit -m "Updated training parameters"

# Push
git push origin main
```

### Pull Latest Changes (in Colab)
```python
# Update code from GitHub
!git pull origin main
```

---

## Troubleshooting

### Git: "Repository not found"
**Solution**: Make repository public or use authentication
```powershell
git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Colab: "CUDA out of memory"
**Solution**: Reduce batch size
```python
config['batch_size'] = 32  # or 16
```

### Colab: Dataset not found
**Solution**: Check path in Google Drive
```python
# List Drive contents
!ls "/content/drive/My Drive"
```

### Git: "Permission denied"
**Solution**: Configure Git credentials
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## File Structure After Push

```
YOUR_REPO/
├── train.py                    # Main training script
├── train_notebook.ipynb        # Jupyter notebook version
├── train_colab.ipynb          # Google Colab version
├── config.py                  # Configuration
├── model.py                   # Model definition
├── dataset_loader.py          # Data loading
├── inference.py               # Prediction script
├── requirements.txt           # Dependencies
├── GITHUB_SETUP.md           # This guide
├── OPTIMIZATION_SUMMARY.md   # Performance details
├── NOTEBOOK_QUICKSTART.md    # Quick start guide
└── .gitignore                # Exclude large files
```

---

## Quick Links

- **Create GitHub Repo**: https://github.com/new
- **Google Colab**: https://colab.research.google.com
- **Google Drive**: https://drive.google.com

---

**Need Help?**
- Check [GITHUB_SETUP.md](GITHUB_SETUP.md) for detailed guide
- Check [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for performance info
