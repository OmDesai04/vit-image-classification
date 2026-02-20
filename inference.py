
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
            
            # Generate output path
            image_path_obj = Path(image_path)
            output_path = image_path_obj.parent / f"gradcam_{image_path_obj.stem}.png"
            
            visualizer.visualize_gradcam(
                image_path=str(image_path),
                output_path=str(output_path),
                alpha=0.4
            )
            
            result['gradcam_path'] = str(output_path)
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
    parser.add_argument('--mode', type=str, choices=['single', 'directory', 'table'], default='table',
                       help='Inference mode: single image, directory of images, or prediction table')
    parser.add_argument('--model', type=str, default='outputs/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--dir', type=str, help='Path to image directory (for directory mode)')
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


if __name__ == "__main__":
    # ============================================================================
    # PREDICTION CONFIGURATION - EDIT THESE VALUES
    # ============================================================================
    RUN_MODE = 'single'  # OPTIONS: 'single', 'batch', 'cli'
                         # 'single' - predict one image (set SINGLE_IMAGE_PATH below)
                         # 'batch'  - predict folder of images (set IMAGE_FOLDER below)
                         # 'cli'    - use command line arguments
    
    # Configure your paths here (use r'...' for Windows paths):
    MODEL_PATH = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\mobilevit.pth' 
    CLASS_NAMES_PATH = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\class_names.json'
    MODEL_NAME = 'mobilevit_s'  # MUST match the architecture used during training
    CROP_SIZE = 0  # Set to 0 for inference (cropping only used during training)
    
    # FOR SINGLE IMAGE PREDICTION:
    SINGLE_IMAGE_PATH = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\unused\l=-5\camera_frame_l-5_f4_20250918_104041.png'  # Change to your image path
    SHOW_IMAGE = False  # Display simple image with prediction (set False to only show Grad-CAM)
    SHOW_GRADCAM = True  # Generate and display Grad-CAM heatmap
    
    # FOR BATCH PREDICTION:
    IMAGE_FOLDER = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\unused'  # Change this to your image folder
    OUTPUT_CSV = r'C:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION\pth files\predictions.csv'
    
    # ============================================================================
    
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
    
    else:
        # Use command line arguments
        main()
