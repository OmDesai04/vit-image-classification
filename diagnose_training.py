"""
Quick Training Performance Analysis
Diagnoses issues with accuracy and training speed
"""

import os
import torch
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_dataset(data_root='split_dataset'):
    """Analyze dataset for potential issues"""
    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    total_images = 0
    class_distribution = {}
    
    for split in splits:
        split_path = Path(data_root) / split
        if not split_path.exists():
            print(f"❌ {split} directory not found!")
            continue
            
        split_images = 0
        classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        
        for class_name in classes:
            class_path = split_path / class_name
            images = list(class_path.glob('*.npy')) + list(class_path.glob('*.jpg')) + \
                    list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
            count = len(images)
            split_images += count
            
            if class_name not in class_distribution:
                class_distribution[class_name] = {'train': 0, 'val': 0, 'test': 0}
            class_distribution[class_name][split] = count
        
        total_images += split_images
        print(f"\n{split.upper()} SET:")
        print(f"  Images: {split_images}")
        print(f"  Classes: {len(classes)}")
        print(f"  Images per class (avg): {split_images/len(classes):.1f}")
    
    print(f"\n📊 TOTAL IMAGES: {total_images}")
    print(f"📁 TOTAL CLASSES: {len(class_distribution)}")
    
    # Check for imbalanced classes
    print("\n⚖️  CLASS BALANCE CHECK:")
    train_counts = [v['train'] for v in class_distribution.values()]
    min_count = min(train_counts)
    max_count = max(train_counts)
    avg_count = np.mean(train_counts)
    
    print(f"  Min images per class: {min_count}")
    print(f"  Max images per class: {max_count}")
    print(f"  Average: {avg_count:.1f}")
    print(f"  Imbalance ratio: {max_count/min_count:.2f}x")
    
    if max_count / min_count > 2:
        print("  ⚠️  WARNING: Significant class imbalance detected!")
    else:
        print("  ✓ Classes are balanced")
    
    # Check for very small classes
    small_classes = [k for k, v in class_distribution.items() if v['train'] < 20]
    if small_classes:
        print(f"\n  ⚠️  {len(small_classes)} classes have < 20 training images!")
        print(f"     This may cause overfitting.")
    
    return class_distribution


def analyze_training_config():
    """Analyze training configuration"""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION ANALYSIS")
    print("="*80)
    
    from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
    
    # Check batch size
    batch_size = DATA_CONFIG['batch_size']
    num_workers = DATA_CONFIG['num_workers']
    
    print(f"\n📦 DATA LOADING:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    if batch_size < 64:
        print("  ⚠️  Batch size is small - try 128 or 256 for faster training")
    if num_workers == 0:
        print("  ⚠️  num_workers=0 - data loading is slow! Set to 4-8")
    
    # Check model
    model_name = MODEL_CONFIG['model_name']
    print(f"\n🤖 MODEL:")
    print(f"  Model: {model_name}")
    
    if 'tiny' in model_name.lower():
        print("  ⚠️  ViT-Tiny may not have enough capacity for 64 classes")
        print("     Consider: vit_small_patch16_224 or vit_base_patch16_224")
    
    # Check training params
    lr = TRAIN_CONFIG['learning_rate']
    use_mixup = TRAIN_CONFIG.get('use_mixup', False)
    label_smoothing = TRAIN_CONFIG.get('label_smoothing', 0)
    
    print(f"\n⚙️  TRAINING:")
    print(f"  Learning rate: {lr}")
    print(f"  Mixup: {use_mixup}")
    print(f"  Label smoothing: {label_smoothing}")
    
    if lr < 1e-4:
        print("  ⚠️  Learning rate is very low - training will be slow")
    if use_mixup and label_smoothing > 0.1:
        print("  ⚠️  Too much regularization (mixup + high label smoothing)")
        print("     May prevent model from learning on small datasets")


def check_GPU():
    """Check GPU availability and memory"""
    print("\n" + "="*80)
    print("GPU CHECK")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"\n✓ CUDA Available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Total Memory: {total_mem:.2f} GB")
        
        # Check current memory usage
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"  Currently Allocated: {allocated:.2f} GB")
        print(f"  Currently Reserved: {reserved:.2f} GB")
        print(f"  Available: {total_mem - reserved:.2f} GB")
        
        if total_mem > 10:
            print("\n  💡 You have plenty of GPU memory!")
            print("     Recommendation: Increase batch_size to 128 or 256")
    else:
        print("\n❌ No GPU detected!")
        print("   Training will be VERY slow on CPU")


def suggest_improvements(class_dist):
    """Suggest specific improvements"""
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    num_classes = len(class_dist)
    train_samples = sum(v['train'] for v in class_dist.values())
    avg_per_class = train_samples / num_classes
    
    print("\n🎯 FOR BETTER ACCURACY:")
    print(f"  1. Use a larger model: vit_small_patch16_224 or vit_base_patch16_224")
    print(f"     (Current: {num_classes} classes with only {avg_per_class:.0f} samples/class)")
    print(f"  2. DISABLE mixup and strong augmentations - dataset is too small")
    print(f"  3. Reduce label_smoothing to 0.05 or disable (0.0)")
    print(f"  4. Increase learning_rate to 5e-4 or 1e-3")
    print(f"  5. Train for 50-100 epochs (not just 30)")
    
    print("\n⚡ FOR FASTER TRAINING:")
    print(f"  1. Increase batch_size to 128 or 256 (you have 16GB GPU)")
    print(f"  2. Set num_workers to 8 for faster data loading")
    print(f"  3. DISABLE torch.compile() - it's slowing you down")
    print(f"  4. Use minimal augmentation (just resize + flip)")
    print(f"  5. Keep AMP enabled (mixed precision)")
    
    print("\n📝 QUICK FIX CONFIG:")
    print("""
# In config.py:
DATA_CONFIG = {
    'batch_size': 128,  # 2x faster
    'num_workers': 8,   # Faster loading
}

MODEL_CONFIG = {
    'model_name': 'vit_small_patch16_224',  # Better capacity
    'use_compile': False,  # Faster startup
}

TRAIN_CONFIG = {
    'epochs': 50,
    'learning_rate': 1e-3,      # Faster learning
    'max_lr': 5e-3,
    'label_smoothing': 0.05,    # Less aggressive
    'dropout': 0.0,             # Disable
    'use_mixup': False,         # Disable for small dataset
}
    """)


if __name__ == "__main__":
    print("\n🔍 TRAINING PERFORMANCE DIAGNOSTIC\n")
    
    # Run analyses
    class_dist = analyze_dataset()
    analyze_training_config()
    check_GPU()
    suggest_improvements(class_dist)
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80 + "\n")
