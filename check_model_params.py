"""
Check PURE Vision Transformer (ViT) models only
"""
import timm
import torch

print("="*60)
print("CHECKING PURE VIT MODELS IN TIMM")
print("="*60)

# Get all ViT models
all_models = timm.list_models('vit*')
print(f"\nFound {len(all_models)} ViT models\n")

# Test small ViT variants
models_to_test = [
    'vit_tiny_patch16_224',
    'vit_tiny_patch8_224',
    'vit_small_patch32_224',
    'vit_small_patch16_224',
    'vit_small_patch8_224',
    'vit_base_patch32_224',
    'vit_base_patch16_224',
]

print(f"{'Model Name':<35} {'Parameters':>15} {'Type':>15}")
print("-" * 70)

for model_name in models_to_test:
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=65)
        total_params = sum(p.numel() for p in model.parameters())
        param_str = f"{total_params/1e6:.2f}M"
        
        # Check if smaller than MobileViT (5-6M)
        if total_params < 6e6:
            type_str = "✓ SMALLER"
        else:
            type_str = "Too large"
        
        print(f"{model_name:<35} {param_str:>15} {type_str:>15}")
    except Exception as e:
        print(f"{model_name:<35} {'ERROR':>15} {'✗':>15}")

print("\n" + "="*70)
print("NOTE: MobileViT baseline = ~5-6M params (99.92% accuracy)")
print("="*70)

