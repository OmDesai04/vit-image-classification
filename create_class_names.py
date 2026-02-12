"""
Create class_names.json file from dataset structure
"""

import json
from pathlib import Path


def create_class_names_json(data_root='split_dataset', output_path='outputs/class_names.json'):
    """
    Extract class names from dataset structure and save to JSON
    
    Args:
        data_root: Path to dataset root (should have train/val/test folders)
        output_path: Where to save class_names.json
    """
    data_root = Path(data_root)
    train_dir = data_root / 'train'
    
    if not train_dir.exists():
        raise ValueError(f"Training directory not found at {train_dir}")
    
    # Get class names from folder names
    class_folders = sorted([
        d.name for d in train_dir.iterdir() 
        if d.is_dir() and d.name != 'unused'
    ])
    
    if not class_folders:
        raise ValueError(f"No class folders found in {train_dir}")
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(class_folders, f, indent=2)
    
    print(f"✓ Created class_names.json with {len(class_folders)} classes")
    print(f"  Saved to: {output_path}")
    print(f"  Classes: {class_folders}")
    
    return class_folders


if __name__ == "__main__":
    create_class_names_json()
