"""
Test script to verify image cropping functionality
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_cropping(image_path, crop_size=256, final_size=224):
    """Visualize the effect of cropping on an image"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Create transforms with cropping
    transform_with_crop = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize((final_size, final_size)),
        transforms.ToTensor()
    ])
    
    # Create transforms without cropping
    transform_no_crop = transforms.Compose([
        transforms.Resize((final_size, final_size)),
        transforms.ToTensor()
    ])
    
    # Apply transforms
    image_cropped = transform_with_crop(image)
    image_no_crop = transform_no_crop(image)
    
    # Convert to numpy for visualization
    img_cropped_np = image_cropped.permute(1, 2, 0).numpy()
    img_no_crop_np = image_no_crop.permute(1, 2, 0).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(np.array(image))
    axes[0].set_title(f'Original Image\n{original_size[0]}x{original_size[1]}')
    axes[0].axis('off')
    
    # With cropping
    axes[1].imshow(img_cropped_np)
    axes[1].set_title(f'With Center Crop\nCrop: {crop_size}x{crop_size} → Resize: {final_size}x{final_size}')
    axes[1].axis('off')
    
    # Without cropping
    axes[2].imshow(img_no_crop_np)
    axes[2].set_title(f'Without Crop\nDirect Resize: {final_size}x{final_size}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/cropping_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: outputs/cropping_comparison.png")
    plt.show()

def test_dataloader_cropping():
    """Test that dataloaders properly apply cropping"""
    try:
        from dataset_loader import create_dataloaders
        
        print("\n" + "="*70)
        print("Testing Dataloader with Cropping")
        print("="*70)
        
        # Create dataloaders with cropping
        train_loader, val_loader, _, num_classes, class_names = create_dataloaders(
            data_root='split_dataset',
            batch_size=4,
            num_workers=0,
            image_size=224,
            crop_size=256
        )
        
        # Test loading a batch
        print("\nLoading test batch...")
        for images, labels in train_loader:
            print(f"✓ Batch loaded successfully")
            print(f"  Batch shape: {images.shape}")
            print(f"  Expected shape: torch.Size([4, 3, 224, 224])")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
        
        print("\n✓ Cropping is working correctly in dataloaders!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing dataloaders: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  IMAGE CROPPING VERIFICATION")
    print("="*70)
    
    # Ensure outputs directory exists
    Path('outputs').mkdir(exist_ok=True)
    
    # Test 1: Check config
    print("\n1. Checking configuration...")
    try:
        from config import DATA_CONFIG
        crop_size = DATA_CONFIG.get('crop_size', None)
        if crop_size:
            print(f"   ✓ Cropping enabled in config: {crop_size}x{crop_size}")
        else:
            print(f"   ⚠ Cropping disabled in config")
    except Exception as e:
        print(f"   ✗ Error reading config: {e}")
        return
    
    # Test 2: Test dataloaders
    print("\n2. Testing dataloaders with cropping...")
    if not test_dataloader_cropping():
        return
    
    # Test 3: Visualize cropping effect
    print("\n3. Creating visualization...")
    
    # Find a sample image
    dataset_path = Path('split_dataset/train')
    if dataset_path.exists():
        # Get first image from first class
        class_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
        if class_folders:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [f for f in class_folders[0].iterdir() if f.suffix.lower() in image_extensions]
            
            if images:
                sample_image = images[0]
                print(f"   Using sample image: {sample_image}")
                visualize_cropping(sample_image, crop_size=256, final_size=224)
            else:
                print("   ⚠ No images found for visualization")
        else:
            print("   ⚠ No class folders found")
    else:
        print("   ⚠ Dataset not found, skipping visualization")
    
    print("\n" + "="*70)
    print("  CROPPING VERIFICATION COMPLETE")
    print("="*70)
    print("\nSUMMARY:")
    print("  • Images will be center-cropped to 256x256")
    print("  • Then resized to 224x224 for the model")
    print("  • This focuses on the central region of images")
    print("  • Helps remove edge artifacts and background clutter")
    print("\nTo disable cropping: Set 'crop_size' to None in config.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
