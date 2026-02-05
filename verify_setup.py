
import sys
from pathlib import Path

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_python_version():
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        return False

def check_dependencies():
    print_section("Checking Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'PyTorch Image Models (timm)',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            if package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {name:30} installed")
        except ImportError:
            print(f"✗ {name:30} NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\n⚠ Install missing packages with: pip install -r requirements.txt")
    
    return all_installed

def check_pytorch_cuda():
    print_section("Checking PyTorch CUDA Support")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA is not available - will use CPU (training will be slower)")
            print("  To enable GPU support, install PyTorch with CUDA:")
            print("  Visit https://pytorch.org/ for installation instructions")
            return False
    except ImportError:
        print("✗ PyTorch is not installed")
        return False

def check_dataset_structure():
    print_section("Checking Dataset Structure")
    
    data_root = Path('split_dataset')
    
    if not data_root.exists():
        print(f"✗ Dataset root not found: {data_root}")
        print("  Make sure 'split_dataset' folder exists in the current directory")
        return False
    
    print(f"✓ Dataset root found: {data_root}")
    
    required_folders = ['train', 'val', 'test']
    all_exist = True
    
    for folder in required_folders:
        folder_path = data_root / folder
        if folder_path.exists():
            class_folders = [d for d in folder_path.iterdir() if d.is_dir() and d.name != 'unused']
            num_classes = len(class_folders)
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            image_count = 0
            for class_folder in class_folders:
                images = [f for f in class_folder.iterdir() if f.suffix.lower() in image_extensions]
                image_count += len(images)
            
            print(f"✓ {folder:10} found: {num_classes} classes, {image_count} images")
        else:
            print(f"✗ {folder:10} NOT found: {folder_path}")
            all_exist = False
    
    return all_exist

def check_project_files():
    print_section("Checking Project Files")
    
    required_files = [
        'dataset_loader.py',
        'model.py',
        'train.py',
        'evaluate.py',
        'inference.py',
        'config.py',
        'requirements.txt'
    ]
    
    all_exist = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} NOT found")
            all_exist = False
    
    return all_exist

def test_dataset_loading():
    print_section("Testing Dataset Loading")
    
    try:
        from dataset_loader import create_dataloaders
        
        print("Attempting to load datasets...")
        train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
            data_root='split_dataset',
            batch_size=4,
            num_workers=0,
            image_size=224
        )
        
        print(f"\n✓ Dataset loading successful!")
        print(f"  Number of classes: {num_classes}")
        print(f"  Sample classes: {class_names[:5]}...")
        
        print("\nTesting batch loading...")
        for images, labels in train_loader:
            print(f"✓ Batch loaded successfully")
            print(f"  Batch shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            break
        
        return True
    
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_model_creation():
    print_section("Testing Model Creation")
    
    try:
        from model import create_model
        
        print("Attempting to create model...")
        model = create_model(
            num_classes=65,
            model_name='vit_base_patch16_224',
            pretrained=False,
            freeze_backbone=False
        )
        
        print(f"\n✓ Model creation successful!")
        
        import torch
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print(" VISION TRANSFORMER SETUP VERIFICATION")
    print("="*60)
    
    results = {}
    
    results['python'] = check_python_version()
    results['dependencies'] = check_dependencies()
    results['cuda'] = check_pytorch_cuda()
    results['dataset'] = check_dataset_structure()
    results['files'] = check_project_files()
    
    if results['dependencies']:
        results['dataset_loading'] = test_dataset_loading()
        results['model_creation'] = test_model_creation()
    
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if all(results.values()):
        print("\n✅ All checks passed! You're ready to start training.")
        print("\nNext step: Run 'python train.py' to start training")
    else:
        print("\n⚠ Some checks failed. Please address the issues above before training.")
        
        if not results.get('dependencies', True):
            print("\n📦 Install dependencies: pip install -r requirements.txt")
        
        if not results.get('dataset', True):
            print("\n📁 Make sure your dataset is in the 'split_dataset' folder")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
