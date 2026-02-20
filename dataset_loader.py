import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class RandomCornerCrop:
    """
    Aggressively crop from random corners (all 4 corners, NOT center).
    This removes edge information and forces model to learn from partial views.
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size
        
    def __call__(self, img):
        w, h = img.size
        crop_size = min(self.crop_size, w, h)
        
        # Only 4 corners - NO center crop to make it harder
        position = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        
        if position == 'top_left':
            left, top = 0, 0
        elif position == 'top_right':
            left, top = w - crop_size, 0
        elif position == 'bottom_left':
            left, top = 0, h - crop_size
        else:  # bottom_right
            left, top = w - crop_size, h - crop_size
        
        return img.crop((left, top, left + crop_size, top + crop_size))


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
            
            # Support both regular images and .npy files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy'}
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
        
        # Check if file is .npy format
        if img_path.suffix.lower() == '.npy':
            # Load numpy array
            image_array = np.load(img_path)
            
            # Handle different numpy array formats
            # Case 1: Float array (0-1 range)
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            # Case 2: Already uint8
            elif image_array.dtype == np.uint8:
                pass
            # Case 3: Other integer types
            else:
                image_array = image_array.astype(np.uint8)
            
            # Ensure correct shape (H, W, C)
            if image_array.ndim == 2:
                # Grayscale - convert to RGB by stacking
                image_array = np.stack([image_array, image_array, image_array], axis=-1)
            elif image_array.ndim == 3:
                # Check if channels are first (C, H, W) and transpose if needed
                if image_array.shape[0] in [1, 3] and image_array.shape[0] < image_array.shape[1]:
                    image_array = np.transpose(image_array, (1, 2, 0))
                # If single channel, convert to RGB
                if image_array.shape[2] == 1:
                    image_array = np.repeat(image_array, 3, axis=2)
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_array)
        else:
            # Regular image loading
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


def get_transforms(image_size=224, is_training=True, crop_size=None):
    """
    Get image transforms with center cropping to focus on central regions.
    
    Args:
        image_size: Final image size for the model
        is_training: Whether to apply training augmentations
        crop_size: Size to center crop (removes edge information, focuses on center)
    """
    if is_training:
        transform_list = []
        
        # Add CENTER crop to focus on center regions (both train and val use center)
        if crop_size is not None and crop_size > 0:
            transform_list.append(transforms.CenterCrop(crop_size))
        
        # Resize and augmentations
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.4),  # Increased
            transforms.RandomRotation(degrees=50),  # Very aggressive
            transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(0.75, 1.25)),  # Very aggressive
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),  # Add perspective distortion
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5, hue=0.25),  # Very strong
            transforms.RandomGrayscale(p=0.25),  # Increased
            transforms.RandomInvert(p=0.15),  # Add random inversion
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 4.0)),  # Very strong blur
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.35)),  # Very aggressive erasing
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        transform = transforms.Compose(transform_list)
    else:
        transform_list = []
        
        # Add center crop if specified
        if crop_size is not None and crop_size > 0:
            transform_list.append(transforms.CenterCrop(crop_size))
        
        # Resize and normalization
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        transform = transforms.Compose(transform_list)
    
    return transform


def create_dataloaders(data_root='split_dataset', 
                      batch_size=32, 
                      num_workers=4,
                      image_size=224,
                      crop_size=None):
    data_root = Path(data_root)
    
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    test_dir = data_root / 'test'
    
    for dir_path, name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if not dir_path.exists():
            raise ValueError(f"{name} directory not found at {dir_path}")
    
    train_transform = get_transforms(image_size=image_size, is_training=True, crop_size=crop_size)
    val_transform = get_transforms(image_size=image_size, is_training=False, crop_size=crop_size)
    
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
    if crop_size is not None and crop_size > 0:
        edge_removal_pct = int((1 - (crop_size/224)**2) * 100)
        print(f"Center cropping enabled: {crop_size}x{crop_size}")
        print(f"  - All images center cropped before augmentations")
        print(f"  - Edge removal: ~{edge_removal_pct}% of image area removed")
        print(f"  - Model focuses on CENTER regions only")
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
        image_size=224,
        crop_size=256
    )
    
    print("\nTesting data loading...")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("\nDataset loader test completed successfully!")
