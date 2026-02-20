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
    'epochs': 20,
    'learning_rate': 1e-5,  # Very low LR
    'weight_decay': 0.25,  # High weight decay
    'scheduler': 'plateau',
    'early_stopping_patience': 7,
    'label_smoothing': 0.45,  # Balanced label smoothing (45%)
    'dropout': 0.55,  # Balanced dropout (55%)
    'use_mixup': True,
    'mixup_alpha': 0.9,  # Strong mixup
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
