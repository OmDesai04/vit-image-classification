DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,  # Smaller batch improves generalization for this dataset
    'num_workers': 4,  # Parallel data loading
    'pin_memory': True,  # Faster CPU to GPU transfer
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 2,  # Prefetch batches for efficiency
    'crop_size': None,  # DISABLED - just resize directly
    'image_extensions': ['.jpg', '.jpeg', '.png'],  # Include PNG to match current dataset
}

MODEL_CONFIG = {
    'model_name': 'swin_tiny_patch4_window7_224',  # Swin Transformer default
    'pretrained': True,
    'freeze_backbone': False,
    'use_compile': False,  # DISABLE - causing slowdown, use standard eager mode
}

TRAIN_CONFIG = {
    'epochs': 60,
    'learning_rate': 2e-4,  # Lower LR for stable Swin fine-tuning
    'max_lr': 8e-4,  # Conservative peak LR
    'weight_decay': 0.05,
    'scheduler': 'onecycle',
    'early_stopping_patience': 10,
    'label_smoothing': 0.1,
    'dropout': 0.3,
    'use_mixup': True,
    'mixup_alpha': 0.2,
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
