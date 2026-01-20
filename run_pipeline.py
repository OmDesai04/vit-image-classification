
import subprocess
import sys
from pathlib import Path
import time

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(command, description):
    print_header(description)
    print(f"Running: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def check_prerequisites():
    print_header("Checking Prerequisites")
    
    if not Path('split_dataset').exists():
        print("✗ Dataset not found: split_dataset/")
        print("  Please ensure the dataset is in the correct location")
        return False
    
    print("✓ Dataset found")
    
    required_files = ['dataset_loader.py', 'model.py', 'train.py', 'evaluate.py', 'inference.py']
    for file in required_files:
        if not Path(file).exists():
            print(f"✗ Required file not found: {file}")
            return False
    
    print("✓ All required files present")
    
    try:
        import torch
        import torchvision
        import timm
        print("✓ Core packages installed")
    except ImportError as e:
        print(f"✗ Required package not installed: {e}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("\n" + "="*70)
    print("  VISION TRANSFORMER IMAGE CLASSIFICATION - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Verify setup")
    print("  2. Train the model")
    print("  3. Evaluate on test set")
    print("  4. Generate prediction table")
    print("\nEstimated time: 1-2 hours (with GPU) or 8-12 hours (CPU)")
    
    response = input("\nProceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Pipeline cancelled.")
        return
    
    start_time = time.time()
    
    if not check_prerequisites():
        print("\n" + "="*70)
        print("  PIPELINE ABORTED - Prerequisites not met")
        print("="*70)
        return
    
    print("\nRunning setup verification...")
    verify_success = run_command(
        "python verify_setup.py",
        "Setup Verification"
    )
    
    if not verify_success:
        response = input("\nVerification had issues. Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Pipeline cancelled.")
            return
    
    train_success = run_command(
        "python train.py",
        "Model Training"
    )
    
    if not train_success:
        print("\n" + "="*70)
        print("  PIPELINE STOPPED - Training failed")
        print("="*70)
        return
    
    eval_success = run_command(
        "python evaluate.py",
        "Model Evaluation"
    )
    
    if not eval_success:
        print("\n" + "="*70)
        print("  PIPELINE STOPPED - Evaluation failed")
        print("="*70)
        return
    
    inference_success = run_command(
        "python inference.py --mode table",
        "Prediction Table Generation"
    )
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print_header("PIPELINE COMPLETE")
    
    if train_success and eval_success and inference_success:
        print("✅ All steps completed successfully!")
        print(f"\n⏱  Total time: {hours}h {minutes}m")
        print("\n📁 Output files generated in 'outputs/' directory:")
        print("   - best_model.pth (trained model)")
        print("   - training_curves.png (training plots)")
        print("   - confusion_matrix.png (confusion matrix)")
        print("   - test_metrics.json (performance metrics)")
        print("   - predictions.csv (prediction table)")
        
        print("\n🎯 Next steps:")
        print("   - Review confusion_matrix.png to identify problem classes")
        print("   - Check test_metrics.json for detailed performance")
        print("   - Use inference.py for predicting new images")
    else:
        print("⚠️  Some steps completed with issues")
        print("   Please review the output above for details")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
