DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'crop_size': None,  # NO cropping - using full images with patch32
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch32_224',  # ViT-Base Patch32 (88M params) - Larger patches = less details
    'pretrained': True,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 3e-5,  # Very low LR for base model
    'weight_decay': 0.3,  # Very high weight decay
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
    'label_smoothing': 0.7,  # EXTREME label smoothing (70%)
    'dropout': 0.8,  # MAXIMUM dropout (80%)
    'use_mixup': True,
    'mixup_alpha': 1.0,  # Maximum mixup
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
