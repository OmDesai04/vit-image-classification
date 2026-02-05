import torch
import torch.nn as nn
import timm


class VisionTransformerClassifier(nn.Module):
    
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True):
        super(VisionTransformerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3  # Add dropout for regularization
        )
        
        print(f"\n{'='*60}")
        print(f"MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Pretrained: {pretrained}")
        print(f"Number of classes: {num_classes}")
        print(f"Input size: 224x224x3")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        return self.vit(x)
    
    def freeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if hasattr(self.vit, 'head'):
            for param in self.vit.head.parameters():
                param.requires_grad = True
        elif hasattr(self.vit, 'fc'):
            for param in self.vit.fc.parameters():
                param.requires_grad = True
        
        print("Backbone frozen. Only classification head will be trained.")
    
    def unfreeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = True
        
        print("All layers unfrozen. Full model will be trained.")
    
    def get_num_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params


def create_model(num_classes, model_name='vit_base_patch16_224', pretrained=True, freeze_backbone=False):
    model = VisionTransformerClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    trainable_params, total_params = model.get_num_params()
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%\n")
    
    return model


def load_model(checkpoint_path, num_classes, model_name='vit_base_patch16_224', device='cuda'):
    model = VisionTransformerClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_accuracy' in checkpoint:
            print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path} (legacy format)")
    
    model.to(device)
    model.eval()
    
    return model


AVAILABLE_MODELS = {
    'vit_tiny_patch16_224': 'ViT-Tiny (5.7M params) - Fastest',
    'vit_small_patch16_224': 'ViT-Small (22M params) - Good balance',
    'vit_base_patch16_224': 'ViT-Base (86M params) - Recommended',
    'vit_large_patch16_224': 'ViT-Large (304M params) - Most accurate',
}


if __name__ == "__main__":
    print("Testing Vision Transformer model creation...\n")
    
    num_classes = 65
    model = create_model(
        num_classes=num_classes,
        model_name='vit_base_patch16_224',
        pretrained=True,
        freeze_backbone=False
    )
    
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output contains logits (before softmax)")
    
    probabilities = torch.softmax(output, dim=1)
    print(f"Probabilities sum: {probabilities.sum(dim=1)}")
    
    print("\nModel test completed successfully!")
