DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 64,  # INCREASED for faster training (adjust based on GPU memory)
    'num_workers': 4,  # Parallel data loading (set to 2-4 for better performance)
    'pin_memory': True,  # Faster CPU to GPU transfer
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 2,  # Prefetch batches for efficiency
    'crop_size': None,  # DISABLED - just resize 1024x1224 to 224x224 directly
}

MODEL_CONFIG = {
    'model_name': 'vit_tiny_patch16_224',  # ViT-Tiny (5M params) - MUCH FASTER than Base (86M)
    'pretrained': True,
    'freeze_backbone': False,
    'use_compile': True,  # PyTorch 2.0+ compile for ~30% speedup
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 3e-4,  # Optimal for ViT with OneCycleLR
    'max_lr': 3e-3,  # Maximum learning rate for OneCycleLR
    'weight_decay': 0.05,  # Stronger regularization
    'scheduler': 'onecycle',  # OneCycleLR for faster convergence
    'early_stopping_patience': 10,
    'label_smoothing': 0.1,  # Helps with generalization
    'dropout': 0.1,  # Light dropout
    'use_mixup': True,  # Data augmentation for better accuracy
    'mixup_alpha': 0.2,
    'use_amp': True,  # Automatic Mixed Precision - 2-3x speedup
    'gradient_clip': 1.0,  # Prevent gradient explosion
    'warmup_epochs': 3,  # Warmup for stable training
}

OUTPUT_CONFIG = {
    'output_dir': 'outputs',
    'save_best_only': True,
    'save_every_n_epochs': 5,
}

INFERENCE_CONFIG = {
    'model_path': 'outputs/best_model.pth',
    'class_names_path': 'outputs/class_names.json',
    'default_output': 'outputs/predictions.csv',
}

DEVICE_CONFIG = {
    'use_cuda': True,
    'device_id': 0,
}
