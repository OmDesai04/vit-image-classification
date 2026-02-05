"""
Verify GhostNet_050 availability and parameters
"""
import timm
import torch

print("="*60)
print("VERIFYING GHOSTNET_050")
print("="*60)

# Check if model exists in timm
all_models = timm.list_models('*ghost*')
print("\nAll GhostNet models available in timm:")
for m in all_models:
    print(f"  ✓ {m}")

print("\n" + "-"*60)
print("Testing ghostnet_050 specifically:\n")

try:
    # Create model
    model = timm.create_model('ghostnet_050', pretrained=False, num_classes=65)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model: ghostnet_050")
    print(f"✓ Total Parameters: {total_params/1e6:.2f}M")
    print(f"✓ Trainable Parameters: {trainable_params/1e6:.2f}M")
    print(f"✓ Status: AVAILABLE IN TIMM")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Forward pass successful: {output.shape}")
    
    print("\n" + "="*60)
    print("✅ GHOSTNET_050 IS READY TO USE!")
    print("="*60)
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    print("\n" + "="*60)
    print("❌ GHOSTNET_050 NOT AVAILABLE")
    print("="*60)

