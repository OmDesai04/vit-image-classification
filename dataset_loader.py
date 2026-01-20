import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageClassificationDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, exclude_folders=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.exclude_folders = exclude_folders or []
        
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        class_folders = sorted([
            d for d in self.root_dir.iterdir() 
            if d.is_dir() and d.name not in self.exclude_folders
        ])
        
        if not class_folders:
            raise ValueError(f"No valid class folders found in {self.root_dir}")
        
        self.class_names = [folder.name for folder in class_folders]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        for class_folder in class_folders:
            class_name = class_folder.name
            class_idx = self.class_to_idx[class_name]
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            image_files = [
                f for f in class_folder.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_distribution(self):
        distribution = {}
        for class_name, class_idx in self.class_to_idx.items():
            count = self.labels.count(class_idx)
            distribution[class_name] = count
        return distribution


def get_transforms(image_size=224, is_training=True):
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(data_root='split_dataset', 
                      batch_size=32, 
                      num_workers=4,
                      image_size=224):
    data_root = Path(data_root)
    
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    test_dir = data_root / 'test'
    
    for dir_path, name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if not dir_path.exists():
            raise ValueError(f"{name} directory not found at {dir_path}")
    
    train_transform = get_transforms(image_size=image_size, is_training=True)
    val_transform = get_transforms(image_size=image_size, is_training=False)
    
    train_dataset = ImageClassificationDataset(
        train_dir, 
        transform=train_transform,
        exclude_folders=['unused']
    )
    
    val_dataset = ImageClassificationDataset(
        val_dir, 
        transform=val_transform,
        exclude_folders=['unused']
    )
    
    test_dataset = ImageClassificationDataset(
        test_dir, 
        transform=val_transform,
        exclude_folders=['unused']
    )
    
    assert train_dataset.class_names == val_dataset.class_names == test_dataset.class_names, \
        "Class names mismatch between train/val/test datasets"
    
    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print("="*60 + "\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
        data_root='split_dataset',
        batch_size=16,
        num_workers=0,
        image_size=224
    )
    
    print("\nTesting data loading...")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("\nDataset loader test completed successfully!")
