DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,  # Smaller batch improves generalization for this dataset
    'num_workers': 2,  # Keep low to avoid worker-overcommit warnings on limited CPUs
    'pin_memory': True,  # Faster CPU to GPU transfer
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 2,  # Prefetch batches for efficiency
    'crop_size': None,  # DISABLED - just resize directly
    'image_extensions': ['.png'],  # Train only on PNG image files
    'check_split_overlap': True,  # Detect duplicate samples across train/val/test
    'split_overlap_strict': True,  # Stop training if leakage is detected
}

MODEL_CONFIG = {
    'model_name': 'swin_tiny_patch4_window7_224',  # Swin Transformer default
    'pretrained': True,
    'freeze_backbone': False,
    'use_compile': False,  # DISABLE - causing slowdown, use standard eager mode
}

TRAIN_CONFIG = {
    'epochs': 60,
    'learning_rate': 1.5e-4,  # Stable LR for ViT-Base fine-tuning
    'max_lr': 6e-4,  # Slightly lower peak LR for better generalization
    'weight_decay': 0.05,
    'scheduler': 'onecycle',
    'early_stopping_patience': 10,
    'label_smoothing': 0.15,
    'dropout': 0.4,
    'use_mixup': True,
    'mixup_alpha': 0.4,
    'use_amp': True,  # Keep AMP for speed
    'gradient_clip': 1.0,
    'warmup_epochs': 8,
    'use_class_weights': True,
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
