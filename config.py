DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 128,  # INCREASED - RTX A4000 has 16GB, can handle more
    'num_workers': 4,  # Parallel data loading
    'pin_memory': True,  # Faster CPU to GPU transfer
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 2,  # Prefetch batches for efficiency
    'crop_size': None,  # DISABLED - just resize directly
}

MODEL_CONFIG = {
    'model_name': 'vit_small_patch16_224',  # Upgraded to Small (22M params) - better for 64 classes
    'pretrained': True,
    'freeze_backbone': False,
    'use_compile': False,  # DISABLE - causing slowdown, use standard eager mode
}

TRAIN_CONFIG = {
    'epochs': 50,  # More epochs for convergence
    'learning_rate': 1e-3,  # HIGHER - faster learning
    'max_lr': 5e-3,  # HIGHER peak for OneCycleLR
    'weight_decay': 0.01,  # Reduced - less regularization for small dataset
    'scheduler': 'onecycle',  # OneCycleLR for faster convergence
    'early_stopping_patience': 15,
    'label_smoothing': 0.05,  # REDUCED - was too aggressive
    'dropout': 0.0,  # DISABLED - pretrained model already regularized
    'use_mixup': False,  # DISABLED - too aggressive for this dataset
    'mixup_alpha': 0.2,
    'use_amp': True,  # Keep AMP for speed
    'gradient_clip': 1.0,
    'warmup_epochs': 5,  # Longer warmup
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
