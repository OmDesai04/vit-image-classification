"""
Check actual image sizes in your dataset
"""
from PIL import Image
from pathlib import Path
from collections import Counter

def check_image_sizes(data_root='split_dataset'):
    """Check the actual dimensions of images in the dataset"""
    
    data_root = Path(data_root)
    
    print("="*60)
    print("CHECKING ACTUAL IMAGE SIZES IN YOUR DATASET")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        
        if not split_dir.exists():
            print(f"\n{split.upper()}: Directory not found")
            continue
        
        print(f"\n{split.upper()} SET:")
        print("-" * 60)
        
        sizes = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # Get all images
        for class_folder in split_dir.iterdir():
            if class_folder.is_dir() and class_folder.name != 'unused':
                for img_file in class_folder.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        try:
                            img = Image.open(img_file)
                            sizes.append(img.size)  # (width, height)
                        except:
                            pass
        
        if not sizes:
            print("  No images found")
            continue
        
        # Count unique sizes
        size_counts = Counter(sizes)
        
        print(f"  Total images checked: {len(sizes)}")
        print(f"  Unique sizes found: {len(size_counts)}")
        print(f"\n  Image Dimensions (Width × Height):")
        
        # Show most common sizes
        for size, count in size_counts.most_common(10):
            width, height = size
            percentage = (count / len(sizes)) * 100
            print(f"    {width:4d} × {height:4d}  :  {count:5d} images ({percentage:5.1f}%)")
        
        if len(size_counts) > 10:
            print(f"    ... and {len(size_counts) - 10} more unique sizes")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_image_sizes()
