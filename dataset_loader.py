import os
import hashlib
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
    
    def __init__(self, root_dir, transform=None, exclude_folders=None, image_extensions=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.exclude_folders = exclude_folders or []
        self.image_extensions = {
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in (image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy', '.noy'])
        }
        
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
            
            image_files = [
                f for f in class_folder.iterdir()
                if f.suffix.lower() in self.image_extensions
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
        
        # Check if file is NumPy format
        if img_path.suffix.lower() in {'.npy', '.noy'}:
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
            # Case 3: uint16 - SCALE to 0-255 range properly
            elif image_array.dtype == np.uint16:
                # Normalize to 0-255 range
                img_min, img_max = image_array.min(), image_array.max()
                if img_max > img_min:
                    image_array = ((image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)
            # Case 4: Other integer types
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
    Get image transforms - SIMPLIFIED for better accuracy on small dataset.
    
    Args:
        image_size: Final image size for the model
        is_training: Whether to apply training augmentations
        crop_size: Size to center crop (removes edge information, focuses on center)
    """
    if is_training:
        # Moderate augmentation to reduce overfitting and keep validation accuracy realistic.
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ]
        
        transform = transforms.Compose(transform_list)
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        transform = transforms.Compose(transform_list)
    
    return transform


def _hash_file(path):
    """Compute a fast content signature to detect duplicates across splits."""
    path = Path(path)
    digest = hashlib.md5()

    # For NumPy files, hash shape/dtype and sampled tensor content.
    if path.suffix.lower() in {'.npy', '.noy'}:
        arr = np.load(path, mmap_mode='r')
        flat = arr.reshape(-1)
        sample_size = min(4096, flat.size)
        head = np.asarray(flat[:sample_size])
        tail = np.asarray(flat[-sample_size:]) if flat.size > sample_size else head

        digest.update(str(arr.shape).encode('utf-8'))
        digest.update(str(arr.dtype).encode('utf-8'))
        digest.update(head.tobytes())
        digest.update(tail.tobytes())
        digest.update(str(flat.size).encode('utf-8'))
        return digest.hexdigest()

    # For standard image files, hash file size + start/end chunks.
    size = path.stat().st_size
    with open(path, 'rb') as f:
        head = f.read(64 * 1024)
        if size > 64 * 1024:
            f.seek(max(size - 64 * 1024, 0))
            tail = f.read(64 * 1024)
        else:
            tail = head

    digest.update(head)
    digest.update(tail)
    digest.update(str(size).encode('utf-8'))
    return digest.hexdigest()


def _build_hash_map(image_paths):
    hash_to_paths = {}
    for path in image_paths:
        file_hash = _hash_file(path)
        hash_to_paths.setdefault(file_hash, []).append(str(path))
    return hash_to_paths


def _find_split_overlaps(train_dataset, val_dataset, test_dataset):
    print("Checking split overlap (fast mode)...")
    train_hashes = _build_hash_map(train_dataset.image_paths)
    val_hashes = _build_hash_map(val_dataset.image_paths)
    test_hashes = _build_hash_map(test_dataset.image_paths)

    train_set = set(train_hashes.keys())
    val_set = set(val_hashes.keys())
    test_set = set(test_hashes.keys())

    overlaps = {
        'train_val': sorted(train_set & val_set),
        'train_test': sorted(train_set & test_set),
        'val_test': sorted(val_set & test_set),
    }

    detailed = {}
    for pair_name, shared_hashes in overlaps.items():
        pair_paths = []
        for h in shared_hashes[:5]:
            if pair_name == 'train_val':
                pair_paths.append((train_hashes[h][0], val_hashes[h][0]))
            elif pair_name == 'train_test':
                pair_paths.append((train_hashes[h][0], test_hashes[h][0]))
            else:
                pair_paths.append((val_hashes[h][0], test_hashes[h][0]))
        detailed[pair_name] = pair_paths

    print("Split overlap check completed.")
    return overlaps, detailed


def create_dataloaders(data_root='split_dataset', 
                      batch_size=32, 
                      num_workers=4,
                      image_size=224,
                      crop_size=None,
                      image_extensions=None,
                      check_split_overlap=True,
                      split_overlap_strict=True,
                      pin_memory=True,
                      persistent_workers=True,
                      prefetch_factor=2):
    data_root = Path(data_root)

    available_cpus = os.cpu_count() or 1
    effective_num_workers = min(max(int(num_workers), 0), available_cpus)
    if effective_num_workers != num_workers:
        print(
            f"Adjusted num_workers from {num_workers} to {effective_num_workers} "
            f"based on available CPU cores ({available_cpus})"
        )
    
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
        exclude_folders=['unused'],
        image_extensions=image_extensions
    )
    
    val_dataset = ImageClassificationDataset(
        val_dir, 
        transform=val_transform,
        exclude_folders=['unused'],
        image_extensions=image_extensions
    )
    
    test_dataset = ImageClassificationDataset(
        test_dir, 
        transform=val_transform,
        exclude_folders=['unused'],
        image_extensions=image_extensions
    )

    if len(train_dataset) == 0:
        raise ValueError(
            f"No training images found in {train_dir}. "
            f"Check folder structure and image_extensions={sorted(train_dataset.image_extensions)}"
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"No validation images found in {val_dir}. "
            f"Check folder structure and image_extensions={sorted(val_dataset.image_extensions)}"
        )
    if len(test_dataset) == 0:
        raise ValueError(
            f"No test images found in {test_dir}. "
            f"Check folder structure and image_extensions={sorted(test_dataset.image_extensions)}"
        )

    if check_split_overlap:
        overlaps, detailed_pairs = _find_split_overlaps(train_dataset, val_dataset, test_dataset)
        total_overlap = sum(len(v) for v in overlaps.values())
        if total_overlap > 0:
            overlap_lines = [
                f"train-val duplicates: {len(overlaps['train_val'])}",
                f"train-test duplicates: {len(overlaps['train_test'])}",
                f"val-test duplicates: {len(overlaps['val_test'])}",
            ]
            examples = []
            for pair_name, pair_examples in detailed_pairs.items():
                if pair_examples:
                    left, right = pair_examples[0]
                    examples.append(f"{pair_name} sample: {left} <-> {right}")

            message = (
                "Detected duplicate files across dataset splits, which can inflate validation/test accuracy.\n"
                + "\n".join(overlap_lines)
            )
            if examples:
                message += "\nExamples:\n" + "\n".join(examples)

            if split_overlap_strict:
                raise ValueError(message)
            else:
                print(f"WARNING: {message}")
    
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
    print(f"Num workers: {effective_num_workers}")
    print(f"Pin memory: {pin_memory}")
    print(f"Persistent workers: {persistent_workers if effective_num_workers > 0 else 'N/A'}")
    print(f"Split overlap check: {'enabled' if check_split_overlap else 'disabled'}")
    print("="*60 + "\n")
    
    # Optimize dataloader settings for faster training
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': effective_num_workers,
        'pin_memory': pin_memory,
    }
    
    # Add persistent_workers and prefetch_factor only if num_workers > 0
    if effective_num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
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
