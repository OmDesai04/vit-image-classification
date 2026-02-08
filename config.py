DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    'crop_size': 256,  # Crop to this size before resizing to image_size (set to None to disable cropping)
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch16_224',  # ViT-Base (86.57M params) - Pure ViT with patch16 (better performance)
    'pretrained': True,  # Use pretrained weights
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,  # Increased from 0.01 to prevent overfitting
    'scheduler': 'plateau',
    'early_stopping_patience': 7,  # Reduced from 10 to stop earlier
    'label_smoothing': 0.35,  # Strong smoothing to prevent overconfidence
    'dropout': 0.3,  # Dropout rate for regularization
    'use_mixup': True,  # Enable Mixup augmentation to prevent overfitting
    'mixup_alpha': 0.4,  # Mixup interpolation strength
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
