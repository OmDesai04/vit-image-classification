# 🚀 GitHub Setup & Jupyter/Colab Instructions

## Part 1: Push Changes to GitHub

### Option A: Using Git Commands (Terminal)

```bash
# Navigate to your project directory
cd "c:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Optimized training pipeline with AMP, OneCycleLR, and Jupyter notebook support"

# Add your GitHub repository as remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### Option B: First Time Setup

If you haven't created a repository yet:

1. Go to [GitHub](https://github.com) and create a new repository
2. Copy the repository URL
3. Run these commands:

```bash
cd "c:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION"
git init
git add .
git commit -m "Initial commit: Optimized image classification pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Create .gitignore (Recommended)

Before pushing, create a `.gitignore` file to exclude large files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Model files (large)
*.pth
*.pt
*.ckpt

# Data directories (too large for GitHub)
split_dataset/
unused/
dataset/
data/

# Output files
outputs/
*.csv
*.png
*.jpg
*.jpeg

# Numpy files
*.npy

# IDE
.vscode/
.idea/

# Large model files
pth files/
```

---

## Part 2: Clone Repository in Jupyter Notebook

### A. For Google Colab

Create a new cell at the top of your notebook:

```python
# ========== CELL 1: Clone Repository ==========

# Clone your GitHub repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Change directory to the cloned repo
import os
os.chdir('/content/YOUR_REPO_NAME')

# Verify we're in the right directory
!pwd
!ls -la
```

### B. For Local Jupyter Notebook

```python
# ========== CELL 1: Navigate to Repository ==========

import os
import subprocess

# Clone if not already present
repo_url = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
repo_name = "YOUR_REPO_NAME"

if not os.path.exists(repo_name):
    !git clone {repo_url}

# Change to repository directory
os.chdir(repo_name)
print(f"Current directory: {os.getcwd()}")
```

---

## Part 3: Mount Google Drive in Colab

### Complete Colab Setup with Drive Mount

```python
# ========== CELL 1: Mount Google Drive & Clone Repo ==========

from google.colab import drive
import os
import shutil

# 1. Mount Google Drive
print("📁 Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
print("✅ Google Drive mounted successfully!")

# 2. Clone repository (if needed)
repo_url = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
repo_name = "IMAGE_CLASSIFICATION"

if not os.path.exists(f'/content/{repo_name}'):
    print(f"\n📥 Cloning repository...")
    !git clone {repo_url}
    print("✅ Repository cloned!")
else:
    print(f"\n✅ Repository already exists")

# 3. Change to repository directory
os.chdir(f'/content/{repo_name}')
print(f"\n📂 Current directory: {os.getcwd()}")

# 4. Install required packages
print("\n📦 Installing dependencies...")
!pip install -q timm tqdm scikit-learn seaborn

print("\n🚀 Setup complete! Ready to train.")
```

---

## Part 4: Complete Colab Training Notebook

Here's a complete notebook setup for Google Colab:

### Cell 1: Setup Environment
```python
# Mount Drive and Clone Repo
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Clone repository
repo_url = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
!git clone {repo_url}

# Navigate to repo
os.chdir('/content/IMAGE_CLASSIFICATION')

# Install dependencies
!pip install -q timm tqdm scikit-learn seaborn

print("✅ Setup complete!")
```

### Cell 2: Check GPU
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Cell 3: Upload Dataset to Drive (One-time)
```python
# Option 1: Upload dataset to Google Drive
# Create a folder in Drive: /My Drive/IMAGE_CLASSIFICATION_DATA/split_dataset/
# Upload your train/val/test folders there

# Option 2: Link to Drive location
import shutil

drive_data_path = '/content/drive/My Drive/IMAGE_CLASSIFICATION_DATA/split_dataset'
local_data_path = '/content/IMAGE_CLASSIFICATION/split_dataset'

if os.path.exists(drive_data_path):
    if not os.path.exists(local_data_path):
        # Create symbolic link to avoid copying
        !ln -s "{drive_data_path}" "{local_data_path}"
        print("✅ Dataset linked from Google Drive")
    else:
        print("✅ Dataset already available")
else:
    print("⚠️ Please upload dataset to Google Drive first!")
    print(f"Expected location: {drive_data_path}")
```

### Cell 4: Configure and Train
```python
# Import modules
from dataset_loader import create_dataloaders
from model import create_model
from train import Trainer
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG

# Configure
config = {**DATA_CONFIG, **MODEL_CONFIG, **TRAIN_CONFIG, **OUTPUT_CONFIG}

# Colab optimizations
config['num_workers'] = 2  # Colab works better with 2
config['batch_size'] = 64  # Adjust based on GPU
config['use_amp'] = True
config['epochs'] = 30

# Load data
train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
    config['data_root'],
    config['batch_size'],
    config['num_workers'],
    config['image_size']
)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(
    num_classes,
    config['model_name'],
    config['pretrained'],
    config['freeze_backbone'],
    config.get('dropout', 0.1)
)

# Train
trainer = Trainer(model, train_loader, val_loader, device, config)
trainer.train()
```

### Cell 5: Save Results to Drive
```python
# Copy outputs to Google Drive for permanent storage
import shutil

output_dir = '/content/IMAGE_CLASSIFICATION/outputs'
drive_output = '/content/drive/My Drive/IMAGE_CLASSIFICATION_OUTPUTS'

if os.path.exists(output_dir):
    os.makedirs(drive_output, exist_ok=True)
    shutil.copytree(output_dir, drive_output, dirs_exist_ok=True)
    print(f"✅ Results saved to Google Drive: {drive_output}")
```

---

## Part 5: Working with Dataset Stored in Google Drive

### Option A: Symbolic Link (Faster, Recommended)
```python
# Link dataset from Drive (no copying needed)
drive_data = '/content/drive/My Drive/Datasets/split_dataset'
local_link = '/content/split_dataset'

if not os.path.exists(local_link):
    !ln -s "{drive_data}" "{local_link}"
    print("✅ Dataset linked")

# Update config
config['data_root'] = local_link
```

### Option B: Copy to Local (Faster Training)
```python
# Copy dataset to Colab's local storage (faster but takes time initially)
import shutil

drive_data = '/content/drive/My Drive/Datasets/split_dataset'
local_data = '/content/split_dataset'

if not os.path.exists(local_data):
    print("📥 Copying dataset to local storage...")
    shutil.copytree(drive_data, local_data)
    print("✅ Dataset copied")

config['data_root'] = local_data
```

---

## Part 6: Troubleshooting

### Issue: "Repository not found"
**Solution**: Make sure repository is public or authenticate:
```bash
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Issue: "Files too large for GitHub"
**Solution**: Use Git LFS or exclude from commits:
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.npy"

# Or exclude in .gitignore
```

### Issue: Colab disconnects and loses progress
**Solution**: Save checkpoints to Google Drive regularly:
```python
# In your training loop
if epoch % 5 == 0:
    torch.save(model.state_dict(), 
               f'/content/drive/My Drive/checkpoints/model_epoch_{epoch}.pth')
```

### Issue: Slow data loading in Colab
**Solution**: 
- Copy dataset to local storage first
- Use `num_workers=2` (not higher in Colab)
- Enable `pin_memory=True`

---

## Quick Command Reference

### Git Commands
```bash
# Clone
git clone https://github.com/USERNAME/REPO.git

# Pull latest changes
git pull origin main

# Push changes
git add .
git commit -m "Your message"
git push origin main

# Check status
git status

# View remote
git remote -v
```

### Colab Commands
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
!nvidia-smi

# Install package
!pip install package_name

# List files
!ls -la

# Check disk space
!df -h
```

---

## Summary

1. **Push to GitHub**: Use git commands or GitHub Desktop
2. **Clone in Colab**: `!git clone https://github.com/USER/REPO.git`
3. **Mount Drive**: `drive.mount('/content/drive')`
4. **Link Dataset**: Use symbolic link for speed
5. **Save Outputs**: Copy results back to Drive

**Ready to go! 🚀**
