"""
Check if ViT-Base patch16 and patch32 are available
"""
import timm
import torch

print("="*75)
print("CHECKING VIT-BASE PATCH VARIANTS")
print("="*75)

# Test ViT-Base patch variants and other models
models_to_test = [
    'mobilevit_s',
    'vit_base_patch16_224',
    'vit_tiny_patch16_224',
    'vit_base_patch32_224',
]

print(f"\n{'Model Name':<30} {'Available':>12} {'Parameters':>15} {'Pretrained':>12}")
print("-" * 75)

for model_name in models_to_test:
    try:
        # Try creating model
        model = timm.create_model(model_name, pretrained=False, num_classes=65)
        total_params = sum(p.numel() for p in model.parameters())
        param_str = f"{total_params/1e6:.2f}M"
        
        # Check if pretrained weights exist
        try:
            model_pretrained = timm.create_model(model_name, pretrained=True, num_classes=1000)
            pretrained_status = "✓ Yes"
        except:
            pretrained_status = "✗ No"
        
        print(f"{model_name:<30} {'✓ Yes':>12} {param_str:>15} {pretrained_status:>12}")
        
    except Exception as e:
        print(f"{model_name:<30} {'✗ No':>12} {'-':>15} {'-':>12}")

print("\n" + "="*75)

