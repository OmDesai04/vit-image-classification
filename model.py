"""
Vision Transformer (ViT) Model for Image Classification
This module defines the ViT model with transfer learning capabilities.
"""

import torch
import torch.nn as nn
import timm


class VisionTransformerClassifier(nn.Module):
    """
    Vision Transformer model for image classification.
    Uses a pretrained ViT backbone with a custom classification head.
    """
    
    def __init__(self, num_classes, model_name='vit_base_patch32_224', pretrained=True, dropout=0.5):
        """
        Initialize the Vision Transformer model
        
        Args:
            num_classes (int): Number of output classes
            model_name (str): Name of the ViT model from timm library (default: vit_base_patch32_224)
            pretrained (bool): Whether to use pretrained weights
            dropout (float): Dropout rate for regularization (default: 0.5 for strong regularization)
        """
        super(VisionTransformerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained ViT model from timm with dropout
        # timm provides various ViT variants optimized for different use cases
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout  # Add dropout for regularization
        )
        
        # Extract patch size from model name
        patch_size = 16  # default
        if 'patch32' in model_name:
            patch_size = 32
        elif 'patch16' in model_name:
            patch_size = 16
        
        num_patches = (224 // patch_size) ** 2
        
        print(f"\n{'='*60}")
        print(f"MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Patch Size: {patch_size}x{patch_size} ({'Larger patches = Less details' if patch_size == 32 else 'Smaller patches = More details'})")
        print(f"Number of Patches: {num_patches} ({224//patch_size}x{224//patch_size} grid)")
        print(f"Pretrained: {pretrained}")
        print(f"Number of classes: {num_classes}")
        print(f"Dropout rate: {dropout}")
        print(f"Input size: 224x224x3")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.vit(x)
    
    def freeze_backbone(self):
        """
        Freeze the backbone (all layers except the classification head)
        Useful for initial fine-tuning with limited data
        """
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze the classification head
        if hasattr(self.vit, 'head'):
            for param in self.vit.head.parameters():
                param.requires_grad = True
        elif hasattr(self.vit, 'fc'):
            for param in self.vit.fc.parameters():
                param.requires_grad = True
        
        print("Backbone frozen. Only classification head will be trained.")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning
        """
        for param in self.vit.parameters():
            param.requires_grad = True
        
        print("All layers unfrozen. Full model will be trained.")
    
    def get_num_params(self):
        """
        Get the number of trainable and total parameters
        
        Returns:
            tuple: (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params


def create_model(num_classes, model_name='vit_base_patch32_224', pretrained=True, freeze_backbone=False, dropout=0.5):
    """
    Create and configure a Vision Transformer model
    
    Args:
        num_classes (int): Number of output classes
        model_name (str): Name of the ViT model variant
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone initially
        dropout (float): Dropout rate for regularization (default: 0.5)
        
    Returns:
        VisionTransformerClassifier: Configured model
    """
    model = VisionTransformerClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    # Print parameter count
    trainable_params, total_params = model.get_num_params()
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%\n")
    
    return model


def load_model(checkpoint_path, num_classes, model_name='vit_base_patch32_224', device='cuda', dropout=0.5):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of output classes
        model_name (str): Name of the ViT model variant
        device (str): Device to load the model on
        dropout (float): Dropout rate for regularization (default: 0.5)
        
    Returns:
        VisionTransformerClassifier: Loaded model
    """
    # Create model architecture
    model = VisionTransformerClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False,  # Don't load pretrained weights when loading from checkpoint
        dropout=dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        
        # Print additional checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_accuracy' in checkpoint:
            print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")
    else:
        # Legacy format: direct state dict
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path} (legacy format)")
    
    model.to(device)
    model.eval()
    
    return model


# Available ViT model variants in timm (as of 2026)
AVAILABLE_MODELS = {
    'vit_tiny_patch16_224': 'ViT-Tiny (5.54M params) - Smallest pure ViT (Current)',
    'vit_small_patch16_224': 'ViT-Small (22M params) - Good balance',
    'vit_base_patch16_224': 'ViT-Base (86M params) - High accuracy',
    'vit_large_patch16_224': 'ViT-Large (304M params) - Most accurate',
}


if __name__ == "__main__":
    """
    Test the model creation
    """
    print("Testing Vision Transformer model creation...\n")
    
    # Create a test model
    num_classes = 65  # Example: 65 classes (l=-32 to l=32)
    model = create_model(
        num_classes=num_classes,
        model_name='vit_base_patch16_224',
        pretrained=True,
        freeze_backbone=False
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output contains logits (before softmax)")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"Probabilities sum: {probabilities.sum(dim=1)}")
    
    print("\nModel test completed successfully!")
