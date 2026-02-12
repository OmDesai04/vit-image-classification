"""
Grad-CAM and Attention Heatmap Visualization for Vision Transformers
Shows which regions of the image the model focuses on
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import argparse
from torchvision import transforms

from model import VisionTransformerClassifier
from config import INFERENCE_CONFIG, DATA_CONFIG


class GradCAMVisualizer:
    """Visualize attention and Grad-CAM for Vision Transformers"""
    
    def __init__(self, model_path, class_names_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = DATA_CONFIG['image_size']
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_name = checkpoint.get('model_name', 'vit_tiny_patch16_224')
        
        self.model = VisionTransformerClassifier(
            num_classes=self.num_classes,
            model_name=self.model_name,
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store gradients and activations
        self.gradients = None
        self.activations = None
        
        print(f"✓ Grad-CAM Visualizer Initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.num_classes}")
    
    def _get_transform(self):
        """Image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _denormalize(self, tensor):
        """Denormalize image tensor for visualization"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    def get_attention_map(self, image_path):
        """
        Extract attention weights from ViT model
        Returns attention heatmap
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_original = img.resize((self.image_size, self.image_size))
        
        transform = self._get_transform()
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass and get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        # Get attention weights from the last transformer block
        # For timm ViT models, we need to access the attention weights
        attention_weights = None
        
        # Hook to extract attention
        def get_attention_hook(module, input, output):
            nonlocal attention_weights
            # For ViT, attention is in the transformer blocks
            if hasattr(output, 'attention_probs'):
                attention_weights = output.attention_probs
        
        # Register hook on the last attention layer
        hooks = []
        for name, module in self.model.vit.named_modules():
            if 'attn' in name and 'drop' not in name:
                hook = module.register_forward_hook(get_attention_hook)
                hooks.append(hook)
        
        # Forward pass to get attention
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create attention heatmap
        # Simple approach: use gradient-based activation
        img_tensor.requires_grad = True
        outputs = self.model(img_tensor)
        
        # Get the predicted class score
        pred_score = outputs[0, pred_idx.item()]
        
        # Backward pass
        self.model.zero_grad()
        pred_score.backward()
        
        # Get gradients
        gradients = img_tensor.grad.data
        
        # Create heatmap from gradients
        heatmap = torch.mean(torch.abs(gradients), dim=1).squeeze().cpu().numpy()
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, img_original, self.class_names[pred_idx.item()], confidence.item()
    
    def visualize_gradcam(self, image_path, output_path='gradcam_output.png', alpha=0.4):
        """
        Generate and save Grad-CAM visualization
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            alpha: Transparency of heatmap overlay (0-1)
        """
        print(f"\nProcessing: {image_path}")
        
        # Get attention heatmap
        heatmap, img_original, predicted_class, confidence = self.get_attention_map(image_path)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (self.image_size, self.image_size))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert PIL image to numpy
        img_np = np.array(img_original)
        
        # Overlay heatmap on image
        overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap only
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlayed
        axes[2].imshow(overlayed)
        axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add prediction info
        fig.suptitle(
            f'Prediction: {predicted_class} (Confidence: {confidence*100:.2f}%)',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {output_path}")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {confidence*100:.2f}%")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'visualization_path': output_path
        }
    
    def visualize_batch(self, image_dir, output_dir='gradcam_outputs'):
        """
        Process multiple images and save Grad-CAM visualizations
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        print(f"\nProcessing {len(image_files)} images...")
        print("="*60)
        
        results = []
        for img_file in image_files:
            output_file = output_path / f"gradcam_{img_file.stem}.png"
            result = self.visualize_gradcam(str(img_file), str(output_file))
            result['input_image'] = img_file.name
            results.append(result)
            print("-"*60)
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
        return results


def main():
    # ========================================================================
    # CONFIGURE YOUR IMAGE DIRECTORY HERE (or use command line arguments)
    # ========================================================================
    DEFAULT_IMAGE_DIR = None  # Set to your image directory path, e.g., 'C:/Users/YourName/Images'
    DEFAULT_IMAGE = None      # Set to your single image path, e.g., 'C:/Users/YourName/test.jpg'
    # ========================================================================
    
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for ViT')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE, help='Path to single input image')
    parser.add_argument('--image-dir', type=str, default=DEFAULT_IMAGE_DIR, help='Directory of images to process')
    parser.add_argument('--output', type=str, default='gradcam_output.png',
                       help='Output path for visualization')
    parser.add_argument('--output-dir', type=str, default='gradcam_outputs',
                       help='Output directory for batch processing')
    parser.add_argument('--model', type=str, default=INFERENCE_CONFIG['model_path'],
                       help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, default=INFERENCE_CONFIG['class_names_path'],
                       help='Path to class names JSON')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Heatmap overlay transparency (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    print("\n" + "="*60)
    print("GRAD-CAM ATTENTION VISUALIZATION")
    print("="*60)
    
    # Initialize visualizer
    visualizer = GradCAMVisualizer(
        model_path=args.model,
        class_names_path=args.classes,
        device=args.device
    )
    
    # Process single image or batch
    if args.image:
        visualizer.visualize_gradcam(
            image_path=args.image,
            output_path=args.output,
            alpha=args.alpha
        )
    elif args.image_dir:
        visualizer.visualize_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir
        )
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
