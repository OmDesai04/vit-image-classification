DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    'crop_size': None,  # Disabled cropping for vit_base_patch32_224
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch32_224',  # ViT-Base with patch32 (88M params) - Faster training without cropping
    'pretrained': True,  # Use pretrained weights
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 35,
    'learning_rate': 2e-4,  # Slightly higher LR for vit_base_patch32
    'weight_decay': 0.03,  # Moderate weight decay for ~90% accuracy
    'scheduler': 'plateau',
    'early_stopping_patience': 8,
    'label_smoothing': 0.1,  # Reduced smoothing for better accuracy
    'dropout': 0.2,  # Lighter dropout for vit_base_patch32
    'use_mixup': True,  # Enable Mixup augmentation
    'mixup_alpha': 0.2,  # Lighter mixup for better accuracy
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
