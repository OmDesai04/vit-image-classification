DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,  # Reduced from 4 to 2 to match system recommendation
    'crop_size': None,  # Disabled cropping for vit_base_patch32_224
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch32_224',  # ViT-Base with patch32 (88M params) - Faster training without cropping
    'pretrained': True,  # Use pretrained weights
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 1.5e-4,  # Slightly reduced LR
    'weight_decay': 0.1,  # Strong weight decay to prevent overfitting
    'scheduler': 'plateau',
    'early_stopping_patience': 8,
    'label_smoothing': 0.4,  # Strong smoothing to reduce accuracy below 95%
    'dropout': 0.5,  # Heavy dropout for regularization
    'use_mixup': True,  # Enable Mixup augmentation
    'mixup_alpha': 0.6,  # Strong mixup to prevent overfitting
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
