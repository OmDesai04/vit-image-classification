
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torchvision import transforms

from model import load_model


class ImageClassifier:
    
    def __init__(self, model_path, class_names, model_name='vit_tiny_patch16_224', device='cuda', image_size=224):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.image_size = image_size
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(
            checkpoint_path=model_path,
            num_classes=self.num_classes,
            model_name=model_name,
            device=self.device
        )
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
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
        
        if output_csv:
            self.save_predictions_to_csv(results, output_csv)
        
        return results
    
    def save_predictions_to_csv(self, results, output_path):
        data = {
            'Image Path': [r['image_path'] for r in results],
            'Predicted Class': [r['predicted_class'] for r in results],
            'Confidence': [r['confidence'] for r in results]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"\nPredictions saved to {output_path}")
    
    def predict_with_true_label(self, image_path, true_label):
        result = self.predict(image_path, return_probabilities=True)
        result['true_label'] = true_label
        result['correct'] = (result['predicted_class'] == true_label)
        
        return result


def predict_single_image(model_path, image_path, class_names_path, model_name='vit_tiny_patch16_224'):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    classifier = ImageClassifier(
        model_path=model_path,
        class_names=class_names,
        model_name=model_name
    )
    
    result = classifier.predict(image_path, return_probabilities=True)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {result['image_path']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nTop-5 Predictions:")
    print("-" * 60)
    for i, (cls, prob) in enumerate(result['top5_predictions'], 1):
        print(f"{i}. {cls:<20} {prob*100:>6.2f}%")
    print("="*60 + "\n")
    
    return result


def predict_from_directory(model_path, image_dir, class_names_path, output_csv, model_name='vit_tiny_patch16_224'):
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
        model_name=model_name
    )
    
    results = classifier.predict_batch(image_paths, output_csv=output_csv)
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Results saved to: {output_csv}")
    print("="*60 + "\n")
    
    return results


def create_prediction_table(test_dir, model_path, class_names_path, output_csv, model_name='vit_tiny_patch16_224'):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    classifier = ImageClassifier(
        model_path=model_path,
        class_names=class_names,
        model_name=model_name
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
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    accuracy = (correct_predictions / len(results)) * 100 if results else 0
    
    print("\n" + "="*60)
    print("PREDICTION TABLE GENERATED")
    print("="*60)
    print(f"Total images: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
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
    
    args = parser.parse_args()
    
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
            model_name=args.model_name
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
            model_name=args.model_name
        )
    
    elif args.mode == 'table':
        create_prediction_table(
            test_dir=args.test_dir,
            model_path=args.model,
            class_names_path=args.class_names,
            output_csv=args.output,
            model_name=args.model_name
        )


if __name__ == "__main__":
    main()
