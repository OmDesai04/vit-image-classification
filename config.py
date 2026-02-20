DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'crop_size': 224,  # Center crop to 224x224 (no edge removal, just crop to model size)
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch16_224',  # ViT-Base Patch16 (86M params) - Smaller patches = more details
    'pretrained': True,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 3e-4,  # Standard learning rate for fine-tuning
    'weight_decay': 0.01,  # Light weight decay
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
    'label_smoothing': 0.1,  # Light label smoothing (10% - standard for best accuracy)
    'dropout': 0.1,  # Light dropout (10% - minimal impact on accuracy)
    'use_mixup': False,  # Disable mixup for higher accuracy
    'mixup_alpha': 0.2,  # Lighter mixup if enabled
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
