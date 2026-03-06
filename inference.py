
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import cv2

from model import load_model
from gradcam_visualizer import GradCAMVisualizer


class ImageClassifier:
    
    def __init__(self, model_path, class_names, model_name='vit_base_patch32_224', device='cuda', image_size=224, crop_size=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.image_size = image_size
        self.crop_size = crop_size
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(
            checkpoint_path=model_path,
            num_classes=self.num_classes,
            model_name=model_name,
            device=self.device
        )
        self.model.eval()
        
        # Build transform with optional cropping
        transform_list = []
        if crop_size is not None and crop_size > 0:
            transform_list.append(transforms.CenterCrop(crop_size))
            print(f"Center cropping enabled: {crop_size}x{crop_size}")
        
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(transform_list)
        
        print(f"Classifier ready on {self.device}")
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image)
        
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def preprocess_npy(self, npy_path):
        """
        Load and preprocess a .npy file containing image data.
        Expects the .npy file to contain a numpy array with shape (H, W, 3) or (H, W)
        in the range [0, 255] or [0, 1].
        """
        # Load the numpy array
        image_array = np.load(npy_path)
        
        # Convert to float if not already
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        elif image_array.dtype in [np.float32, np.float64]:
            # Check if values are in [0, 255] range
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
        
        # Handle grayscale images (H, W) -> (H, W, 3)
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        # Ensure it's in the correct shape (H, W, C)
        if image_array.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got {image_array.shape[-1]}")
        
        # Convert to PIL Image for transformation pipeline
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        
        # Apply the same transformations as regular images
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_path, return_probabilities=False):
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        predicted_class = self.class_names[predicted_idx]
        
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'predicted_index': predicted_idx,
            'confidence': confidence
        }
        
        if return_probabilities:
            top5_probs, top5_indices = torch.topk(probabilities[0], k=min(5, self.num_classes))
            top5_classes = [self.class_names[idx] for idx in top5_indices.cpu().numpy()]
            top5_probs = top5_probs.cpu().numpy()
            
            result['top5_predictions'] = list(zip(top5_classes, top5_probs))
            result['all_probabilities'] = probabilities[0].cpu().numpy().tolist()
        
        return result
    
    def predict_npy(self, npy_path, return_probabilities=False):
        """
        Predict from a .npy file containing image data.
        """
        image_tensor = self.preprocess_npy(npy_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        predicted_class = self.class_names[predicted_idx]
        
        result = {
            'npy_path': str(npy_path),
            'predicted_class': predicted_class,
            'predicted_index': predicted_idx,
            'confidence': confidence
        }
        
        if return_probabilities:
            top5_probs, top5_indices = torch.topk(probabilities[0], k=min(5, self.num_classes))
            top5_classes = [self.class_names[idx] for idx in top5_indices.cpu().numpy()]
            top5_probs = top5_probs.cpu().numpy()
            
            result['top5_predictions'] = list(zip(top5_classes, top5_probs))
            result['all_probabilities'] = probabilities[0].cpu().numpy().tolist()
        
        return result
    
    def predict_batch(self, image_paths, output_csv=None):
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths, 1):
            try:
                # Ensure image_path is a Path object for consistent handling
                if not isinstance(image_path, Path):
                    image_path = Path(image_path)
                
                result = self.predict(image_path)
                results.append(result)
                
                if i % 10 == 0 or i == len(image_paths):
                    print(f"Processed {i}/{len(image_paths)} images")
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'predicted_class': 'ERROR',
                    'predicted_index': -1,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        if output_csv and len(results) > 0:
            self.save_predictions_to_csv(results, output_csv)
        
        return results
    
    def predict_batch_npy(self, npy_paths, output_csv=None):
        """
        Batch prediction for .npy files.
        """
        results = []
        
        print(f"\nProcessing {len(npy_paths)} .npy files...")
        for i, npy_path in enumerate(npy_paths, 1):
            try:
                # Ensure npy_path is a Path object for consistent handling
                if not isinstance(npy_path, Path):
                    npy_path = Path(npy_path)
                
                result = self.predict_npy(npy_path)
                results.append(result)
                
                if i % 10 == 0 or i == len(npy_paths):
                    print(f"Processed {i}/{len(npy_paths)} .npy files")
            
            except Exception as e:
                print(f"Error processing {npy_path}: {e}")
                results.append({
                    'npy_path': str(npy_path),
                    'predicted_class': 'ERROR',
                    'predicted_index': -1,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        if output_csv and len(results) > 0:
            self.save_npy_predictions_to_csv(results, output_csv)
        
        return results
    
    def save_predictions_to_csv(self, results, output_path):
        if not results:
            print("No results to save.")
            return
        
        data = {
            'Image Path': [r.get('image_path', '') for r in results],
            'Predicted Class': [r.get('predicted_class', 'ERROR') for r in results],
            'Confidence': [r.get('confidence', 0.0) for r in results]
        }
        
        # Add error column if any errors exist
        if any('error' in r for r in results):
            data['Error'] = [r.get('error', '') for r in results]
        
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"\nPredictions saved to {output_path}")
    
    def save_npy_predictions_to_csv(self, results, output_path):
        """
        Save .npy prediction results to CSV.
        """
        if not results:
            print("No results to save.")
            return
        
        data = {
            'NPY Path': [r.get('npy_path', '') for r in results],
            'Predicted Class': [r.get('predicted_class', 'ERROR') for r in results],
            'Confidence': [r.get('confidence', 0.0) for r in results]
        }
        
        # Add error column if any errors exist
        if any('error' in r for r in results):
            data['Error'] = [r.get('error', '') for r in results]
        
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"\nPredictions saved to {output_path}")
    
    def predict_with_true_label(self, image_path, true_label):
        result = self.predict(image_path, return_probabilities=True)
        result['true_label'] = true_label
        result['correct'] = (result['predicted_class'] == true_label)
        
        return result


def predict_single_image(model_path, image_path, class_names_path, model_name='vit_base_patch32_224', 
                        crop_size=None, show_image=True, show_gradcam=True):
    """
    Predict single image with visualization and Grad-CAM
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
        class_names_path: Path to class names JSON
        model_name: Model architecture name
        crop_size: Optional cropping size
        show_image: Display image with prediction overlay
        show_gradcam: Generate and display Grad-CAM heatmap
    """
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    classifier = ImageClassifier(
        model_path=model_path,
        class_names=class_names,
        model_name=model_name,
        crop_size=crop_size
    )
    
    # Start timing
    start_time = time.time()
    
    result = classifier.predict(image_path, return_probabilities=True)
    
    # End timing
    inference_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {result['image_path']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Inference Time: {inference_time:.3f} seconds")
    print("="*60 + "\n")
    
    # Display image with prediction overlay
    if show_image:
        try:
            img = Image.open(image_path).convert('RGB')
            img_display = img.copy()
            
            # Resize for display if too large
            max_size = 800
            if max(img_display.size) > max_size:
                ratio = max_size / max(img_display.size)
                new_size = tuple(int(dim * ratio) for dim in img_display.size)
                img_display = img_display.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.imshow(img_display)
            plt.axis('off')
            
            # Add prediction text
            title_text = f"Predicted: {result['predicted_class']}\n" \
                        f"Confidence: {result['confidence']*100:.2f}%\n" \
                        f"Time: {inference_time:.3f}s"
            plt.title(title_text, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.show(block=False)
            print("✓ Image displayed with prediction overlay")
        except Exception as e:
            print(f"⚠ Could not display image: {e}")
    
    # Generate Grad-CAM heatmap
    if show_gradcam:
        try:
            print("\nGenerating Grad-CAM heatmap...")
            visualizer = GradCAMVisualizer(
                model_path=model_path,
                class_names_path=class_names_path,
                model_name=model_name
            )
            
            # Use temporary path (will be displayed, not saved permanently)
            visualizer.visualize_gradcam(
                image_path=str(image_path),
                output_path="temp_gradcam.png",
                alpha=0.4
            )
        except Exception as e:
            print(f"⚠ Could not generate Grad-CAM: {e}")
    
    # Keep windows open
    if show_image or show_gradcam:
        print("\n[Close the visualization window to continue]")
        plt.show()
    
    return result


def predict_from_directory(model_path, image_dir, class_names_path, output_csv, model_name='vit_base_patch32_224', crop_size=None):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = [f for f in image_dir.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    classifier = ImageClassifier(
        model_path=model_path,
        class_names=class_names,
        model_name=model_name,
        crop_size=crop_size
    )
    
    results = classifier.predict_batch(image_paths, output_csv=output_csv)
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Results saved to: {output_csv}")
    print("="*60 + "\n")
    
    return results


def predict_npy_directory(model_path, npy_dir, class_names_path, output_csv, model_name='vit_base_patch32_224', crop_size=None):
    """
    Batch prediction for all .npy files in a directory.
    
    Args:
        model_path: Path to trained model
        npy_dir: Directory containing .npy files
        class_names_path: Path to class names JSON
        output_csv: Path to save predictions CSV
        model_name: Model architecture name
        crop_size: Optional cropping size
    """
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    npy_dir = Path(npy_dir)
    npy_paths = [f for f in npy_dir.rglob('*.npy')]
    
    if not npy_paths:
        print(f"No .npy files found in {npy_dir}")
        return
    
    print(f"Found {len(npy_paths)} .npy files in {npy_dir}")
    
    classifier = ImageClassifier(
        model_path=model_path,
        class_names=class_names,
        model_name=model_name,
        crop_size=crop_size
    )
    
    start_time = time.time()
    results = classifier.predict_batch_npy(npy_paths, output_csv=output_csv)
    end_time = time.time()
    
    time_taken = end_time - start_time
    files_per_second = len(results) / time_taken if time_taken > 0 else 0
    
    print("\n" + "="*60)
    print("NPY BATCH PREDICTION SUMMARY")
    print("="*60)
    print(f"Total .npy files processed: {len(results)}")
    print(f"Time taken: {time_taken:.2f} seconds ({files_per_second:.2f} files/sec)")
    print(f"Results saved to: {output_csv}")
    print("="*60 + "\n")
    
    # Show sample predictions
    if results:
        print("Sample predictions:")
        print("-" * 80)
        print(f"{'File Name':<40} {'Predicted Class':<20} {'Confidence':<10}")
        print("-" * 80)
        for i, res in enumerate(results[:10]):
            file_name = Path(res.get('npy_path', '')).name
            pred_class = res.get('predicted_class', 'ERROR')
            confidence = res.get('confidence', 0.0)
            print(f"{file_name:<40} {pred_class:<20} {confidence*100:.2f}%")
        print("-" * 80 + "\n")
    
    return results


def create_prediction_table(test_dir, model_path, class_names_path, output_csv, model_name='vit_base_patch32_224', crop_size=None):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    classifier = ImageClassifier(
        model_path=model_path,
        class_names=class_names,
        model_name=model_name,
        crop_size=crop_size
    )
    
    test_dir = Path(test_dir)
    image_data = []
    
    print(f"Scanning test directory: {test_dir}")
    
    for class_folder in sorted(test_dir.iterdir()):
        if class_folder.is_dir() and class_folder.name != 'unused':
            true_label = class_folder.name
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            images = [f for f in class_folder.iterdir() if f.suffix.lower() in image_extensions]
            
            for img_path in images:
                image_data.append((img_path, true_label))
    
    print(f"Found {len(image_data)} images across {len(class_names)} classes")
    
    results = []
    correct_predictions = 0
    
    # Start timing
    start_time = time.time()
    
    print("\nGenerating predictions...")
    for i, (img_path, true_label) in enumerate(image_data, 1):
        try:
            result = classifier.predict_with_true_label(img_path, true_label)
            results.append({
                'Image Name': img_path.name,
                'Image Path': str(img_path),
                'True Label': true_label,
                'Predicted Label': result['predicted_class'],
                'Confidence': f"{result['confidence']*100:.2f}%",
                'Correct': '✓' if result['correct'] else '✗'
            })
            
            if result['correct']:
                correct_predictions += 1
            
            if i % 50 == 0 or i == len(image_data):
                print(f"Processed {i}/{len(image_data)} images")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'Image Name': img_path.name,
                'Image Path': str(img_path),
                'True Label': true_label,
                'Predicted Label': 'ERROR',
                'Confidence': '0.00%',
                'Correct': '✗'
            })
    
    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    
    if not results:
        print("\nNo results to save. Exiting.")
        return
    
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    
    accuracy = (correct_predictions / len(results)) * 100 if results else 0
    images_per_second = len(results) / time_taken if time_taken > 0 else 0
    
    # Format time as "X min Y sec"
    minutes = int(time_taken // 60)
    seconds = int(time_taken % 60)
    if minutes > 0:
        time_str = f"{minutes} min {seconds} sec"
    else:
        time_str = f"{seconds} sec"
    
    print("\n" + "="*60)
    print("PREDICTION TABLE GENERATED")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total images: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time taken: {time_str} ({images_per_second:.2f} images/sec)")
    print(f"Results saved to: {output_csv}")
    print("="*60 + "\n")
    
    print("Sample predictions:")
    print("-" * 80)
    print(f"{'Image Name':<30} {'True Label':<15} {'Predicted Label':<15} {'Result':<10}")
    print("-" * 80)
    for i, row in enumerate(results[:10]):
        print(f"{row['Image Name']:<30} {row['True Label']:<15} {row['Predicted Label']:<15} {row['Correct']:<10}")
    print("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('--mode', type=str, choices=['single', 'directory', 'table', 'npy'], default='table',
                       help='Inference mode: single image, directory of images, prediction table, or npy batch')
    parser.add_argument('--model', type=str, default='outputs/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--dir', type=str, help='Path to image directory (for directory mode)')
    parser.add_argument('--npy-dir', type=str, help='Path to .npy directory (for npy mode)')
    parser.add_argument('--test-dir', type=str, default='split_dataset/test',
                       help='Path to test directory (for table mode)')
    parser.add_argument('--output', type=str, default='outputs/predictions.csv',
                       help='Path to save predictions CSV')
    parser.add_argument('--class-names', type=str, default='outputs/class_names.json',
                       help='Path to class names JSON file')
    parser.add_argument('--model-name', type=str, default='vit_base_patch32_224',
                       help='ViT model variant name')
    parser.add_argument('--crop-size', type=int, default=0,
                       help='Size to crop images before resizing (None or 0 to disable cropping)')
    
    args = parser.parse_args()
    
    # Convert crop_size to None if 0
    crop_size = args.crop_size if args.crop_size > 0 else None
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found at {args.model}")
        print("Please train the model first by running: python train.py")
        return
    
    if not Path(args.class_names).exists():
        print(f"Error: Class names file not found at {args.class_names}")
        print("Please train the model first by running: python train.py")
        return
    
    if args.mode == 'single':
        if not args.image:
            print("Error: --image argument required for single mode")
            return
        
        predict_single_image(
            model_path=args.model,
            image_path=args.image,
            class_names_path=args.class_names,
            model_name=args.model_name,
            crop_size=crop_size
        )
    
    elif args.mode == 'directory':
        if not args.dir:
            print("Error: --dir argument required for directory mode")
            return
        
        predict_from_directory(
            model_path=args.model,
            image_dir=args.dir,
            class_names_path=args.class_names,
            output_csv=args.output,
            model_name=args.model_name,
            crop_size=crop_size
        )
    
    elif args.mode == 'table':
        create_prediction_table(
            test_dir=args.test_dir,
            model_path=args.model,
            class_names_path=args.class_names,
            output_csv=args.output,
            model_name=args.model_name,
            crop_size=crop_size
        )
    
    elif args.mode == 'npy':
        if not args.npy_dir:
            print("Error: --npy-dir argument required for npy mode")
            return
        
        predict_npy_directory(
            model_path=args.model,
            npy_dir=args.npy_dir,
            class_names_path=args.class_names,
            output_csv=args.output,
            model_name=args.model_name,
            crop_size=crop_size
        )


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════════════
    # 📝 EASY CONFIGURATION - JUST EDIT THE PATHS BELOW
    # ═══════════════════════════════════════════════════════════════════════════
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Choose your RUN_MODE
    # ──────────────────────────────────────────────────────────────────────────
    RUN_MODE = 'npy'  # 👈 CHANGE THIS!
                      # OPTIONS:
                      #   'single' → Predict one image
                      #   'batch'  → Predict folder of images (.jpg, .png, etc.)
                      #   'npy'    → Predict folder of .npy files
                      #   'cli'    → Use command line arguments
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Set your MODEL and CLASS NAMES paths
    # ──────────────────────────────────────────────────────────────────────────
    MODEL_PATH = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\mobilevit.pth' 
    CLASS_NAMES_PATH = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\class_names.json'
    MODEL_NAME = 'mobilevit_s'  # MUST match the architecture used during training
    CROP_SIZE = 0  # Set to 0 for inference (cropping only used during training)
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Configure paths based on your RUN_MODE
    # ──────────────────────────────────────────────────────────────────────────
    
    # 🖼️ FOR SINGLE IMAGE PREDICTION (RUN_MODE = 'single'):
    SINGLE_IMAGE_PATH = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\unused\l=-5\camera_frame_l-5_f4_20250918_104041.png'
    SHOW_IMAGE = False  # Show image with prediction overlay
    SHOW_GRADCAM = True  # Show Grad-CAM heatmap
    
    # 📁 FOR BATCH IMAGE PREDICTION (RUN_MODE = 'batch'):
    IMAGE_FOLDER = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\unused'
    OUTPUT_CSV = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\predictions.csv'
    
    # 📊 FOR NPY BATCH PREDICTION (RUN_MODE = 'npy'):
    NPY_FOLDER = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION'  # 👈 EDIT: Folder with .npy files
    NPY_OUTPUT_CSV = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\npy_predictions.csv'  # 👈 EDIT: Output CSV path
    
    # ═══════════════════════════════════════════════════════════════════════════
    
    if RUN_MODE == 'single':
        # SINGLE IMAGE PREDICTION
        print("Running single image prediction...")
        print(f"Model: {MODEL_PATH}")
        print(f"Image: {SINGLE_IMAGE_PATH}\n")
        
        if not Path(MODEL_PATH).exists():
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please update MODEL_PATH in the code.")
            exit(1)
        
        if not Path(CLASS_NAMES_PATH).exists():
            print(f"Error: Class names file not found at {CLASS_NAMES_PATH}")
            print("Please update CLASS_NAMES_PATH in the code.")
            exit(1)
        
        if not Path(SINGLE_IMAGE_PATH).exists():
            print(f"Error: Image file not found at {SINGLE_IMAGE_PATH}")
            print("Please update SINGLE_IMAGE_PATH in the code.")
            exit(1)
        
        crop_size = CROP_SIZE if CROP_SIZE > 0 else None
        
        predict_single_image(
            model_path=MODEL_PATH,
            image_path=SINGLE_IMAGE_PATH,
            class_names_path=CLASS_NAMES_PATH,
            model_name=MODEL_NAME,
            crop_size=crop_size,
            show_image=SHOW_IMAGE,
            show_gradcam=SHOW_GRADCAM
        )
    
    elif RUN_MODE == 'batch':
        # BATCH PREDICTION
        print("Running batch prediction with configured paths...")
        print(f"Model: {MODEL_PATH}")
        print(f"Images: {IMAGE_FOLDER}")
        print(f"Output: {OUTPUT_CSV}\n")
        
        if not Path(MODEL_PATH).exists():
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please update MODEL_PATH in the code.")
            exit(1)
        
        if not Path(CLASS_NAMES_PATH).exists():
            print(f"Error: Class names file not found at {CLASS_NAMES_PATH}")
            print("Please update CLASS_NAMES_PATH in the code.")
            exit(1)
        
        if not Path(IMAGE_FOLDER).exists():
            print(f"Error: Image folder not found at {IMAGE_FOLDER}")
            print("Please update IMAGE_FOLDER in the code.")
            exit(1)
        
        crop_size = CROP_SIZE if CROP_SIZE > 0 else None
        
        # Check if IMAGE_FOLDER has subdirectories (class folders) or just images
        image_folder_path = Path(IMAGE_FOLDER)
        has_subdirs = any(item.is_dir() for item in image_folder_path.iterdir() if item.name != 'unused')
        
        if has_subdirs:
            # Use table mode for organized dataset
            print("Detected class folders - using table mode\n")
            create_prediction_table(
                test_dir=IMAGE_FOLDER,
                model_path=MODEL_PATH,
                class_names_path=CLASS_NAMES_PATH,
                output_csv=OUTPUT_CSV,
                model_name=MODEL_NAME,
                crop_size=crop_size
            )
        else:
            # Use directory mode for flat folder
            print("No class folders detected - using directory mode\n")
            predict_from_directory(
                model_path=MODEL_PATH,
                image_dir=IMAGE_FOLDER,
                class_names_path=CLASS_NAMES_PATH,
                output_csv=OUTPUT_CSV,
                model_name=MODEL_NAME,
                crop_size=crop_size
            )
    
    elif RUN_MODE == 'npy':
        # NPY BATCH PREDICTION
        print("Running .npy batch prediction...")
        print(f"Model: {MODEL_PATH}")
        print(f"NPY Folder: {NPY_FOLDER}")
        print(f"Output: {NPY_OUTPUT_CSV}\n")
        
        if not Path(MODEL_PATH).exists():
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please update MODEL_PATH in the code.")
            exit(1)
        
        if not Path(CLASS_NAMES_PATH).exists():
            print(f"Error: Class names file not found at {CLASS_NAMES_PATH}")
            print("Please update CLASS_NAMES_PATH in the code.")
            exit(1)
        
        if not Path(NPY_FOLDER).exists():
            print(f"Error: NPY folder not found at {NPY_FOLDER}")
            print("Please update NPY_FOLDER in the code.")
            exit(1)
        
        crop_size = CROP_SIZE if CROP_SIZE > 0 else None
        
        predict_npy_directory(
            model_path=MODEL_PATH,
            npy_dir=NPY_FOLDER,
            class_names_path=CLASS_NAMES_PATH,
            output_csv=NPY_OUTPUT_CSV,
            model_name=MODEL_NAME,
            crop_size=crop_size
        )
    
    else:
        # Use command line arguments
        main()
