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
    
    def __init__(self, model_path, class_names_path, model_name='mobilevit_s', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = DATA_CONFIG['image_size']
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            self.model_name = checkpoint.get('model_name', model_name)
            state_dict = checkpoint['model_state_dict']
        else:
            # Legacy format: checkpoint IS the state dict
            self.model_name = model_name
            state_dict = checkpoint
        
        self.model = VisionTransformerClassifier(
            num_classes=self.num_classes,
            model_name=self.model_name,
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
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
        Extract attention weights using proper Grad-CAM
        Returns attention heatmap
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_original = img.resize((self.image_size, self.image_size))
        
        transform = self._get_transform()
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the best layer to hook based on model architecture
        target_layer = None
        
        # Try to find appropriate layer for different architectures
        for name, module in self.model.vit.named_modules():
            # For ViT models: use the last block before classification
            if 'blocks' in name and isinstance(module, nn.Module):
                target_layer = module
            # For MobileViT: use last conv or MV block
            elif 'conv' in name.lower() and isinstance(module, nn.Conv2d):
                target_layer = module
            # For any model: use the last normalization layer
            elif 'norm' in name.lower() and isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                target_layer = module
        
        # If no suitable layer found, use a fallback approach
        if target_layer is None:
            # Use gradient-only approach
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            # Get gradients with respect to input
            pred_score = outputs[0, pred_idx.item()]
            self.model.zero_grad()
            pred_score.backward()
            
            gradients = img_tensor.grad.data.squeeze().cpu()
            # Use absolute gradients and average across channels
            heatmap = torch.mean(torch.abs(gradients), dim=0).numpy()
        else:
            # Register hooks
            forward_handle = target_layer.register_forward_hook(forward_hook)
            backward_handle = target_layer.register_full_backward_hook(backward_hook)
            
            # Forward pass
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            # Backward pass
            pred_score = outputs[0, pred_idx.item()]
            self.model.zero_grad()
            pred_score.backward()
            
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
            
            # Compute Grad-CAM
            if self.gradients is not None and self.activations is not None:
                # Compute weights
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) if len(self.gradients.shape) == 4 else torch.mean(self.gradients, dim=[0, 1])
                activations = self.activations.detach()
                
                # Weight the channels
                if len(activations.shape) == 4:  # Conv feature maps
                    for i in range(activations.shape[1]):
                        activations[:, i, :, :] *= pooled_gradients[i]
                    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
                else:  # Transformer features
                    heatmap = torch.mean(torch.abs(activations), dim=[0, 1]).squeeze().cpu().numpy()
                    # Reshape if needed
                    if heatmap.ndim == 1:
                        size = int(np.sqrt(heatmap.shape[0]))
                        heatmap = heatmap.reshape(size, size)
            else:
                # Fallback to gradients
                gradients = img_tensor.grad.data.squeeze().cpu()
                heatmap = torch.mean(torch.abs(gradients), dim=0).numpy()
        
        # Apply ReLU to focus on positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Enhance contrast using power transform
        heatmap = np.power(heatmap, 0.7)  # Makes mid-range values more visible
        
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
        
        # Create figure with subplots and more space for text
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.15], hspace=0.3, wspace=0.2)
        
        # Original image
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(img_np)
        ax0.set_title('1. Original Image', fontsize=13, fontweight='bold', pad=10)
        ax0.axis('off')
        
        # Heatmap only with colorbar
        ax1 = fig.add_subplot(gs[0, 1])
        im = ax1.imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        ax1.set_title('2. Where the Model Looks\n(Attention Heatmap)', fontsize=13, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Add colorbar to heatmap
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity', rotation=270, labelpad=20, fontsize=10)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.set_yticklabels(['Low\n(Ignored)', '', '', '', 'High\n(Focused)'], fontsize=9)
        
        # Overlayed
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(overlayed)
        ax2.set_title('3. Combined View\n(Overlay)', fontsize=13, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # Add prediction info at top
        fig.suptitle(
            f'Prediction: {predicted_class}  |  Confidence: {confidence*100:.2f}%',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # Use subplots_adjust instead of tight_layout to avoid warnings
        plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05, wspace=0.2, hspace=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        # Don't close - keep it open for display
        # plt.close()
        
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
    # CONFIGURE YOUR PATHS HERE (or use command line arguments)
    # ========================================================================
    DEFAULT_MODEL_PATH = None        # Set to your .pth file path, e.g., 'C:/path/to/best_model.pth'
    DEFAULT_CLASS_NAMES_PATH = None  # Set to your class_names.json path, e.g., 'C:/path/to/class_names.json'
    DEFAULT_IMAGE_DIR = None         # Set to your image directory path, e.g., 'C:/Users/YourName/Images'
    DEFAULT_IMAGE = None             # Set to your single image path, e.g., 'C:/Users/YourName/test.jpg'
    # ========================================================================
    
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for ViT')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE, help='Path to single input image')
    parser.add_argument('--image-dir', type=str, default=DEFAULT_IMAGE_DIR, help='Directory of images to process')
    parser.add_argument('--output', type=str, default='gradcam_output.png',
                       help='Output path for visualization')
    parser.add_argument('--output-dir', type=str, default='gradcam_outputs',
                       help='Output directory for batch processing')
    parser.add_argument('--model', type=str, 
                       default=DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH else INFERENCE_CONFIG['model_path'],
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--classes', type=str, 
                       default=DEFAULT_CLASS_NAMES_PATH if DEFAULT_CLASS_NAMES_PATH else INFERENCE_CONFIG['class_names_path'],
                       help='Path to class names JSON')
    parser.add_argument('--model-name', type=str, default='mobilevit_s',
                       help='Model architecture name')
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
        model_name=args.model_name,
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
