DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 16,  # Reduced for faster epochs
    'num_workers': 4,  # Increased for faster data loading
    'crop_size': 224,  # Center crop to 224x224 (KEEPING as requested)
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch16_224',  # ViT-Base Patch16 (86M params) - Smaller patches = more details
    'pretrained': True,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 1e-3,  # INCREASED - was too low for learning
    'weight_decay': 0.01,
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
    'label_smoothing': 0.0,  # DISABLED - no smoothing for clear learning signal
    'dropout': 0.0,  # DISABLED initially - can add back after model learns
    'use_mixup': False,
    'mixup_alpha': 0.2,
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
