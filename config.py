DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'crop_size': 180,  # Random corner/edge cropping during training (4 corners + center)
}

MODEL_CONFIG = {
    'model_name': 'vit_tiny_patch16_224',  # ViT-Tiny (5.7M params) - 15x smaller to prevent overfitting
    'pretrained': True,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 1e-4,  # Lower LR for tiny model
    'weight_decay': 0.15,  # Very strong weight decay
    'scheduler': 'plateau',
    'early_stopping_patience': 8,
    'label_smoothing': 0.5,  # Very strong label smoothing (50%)
    'dropout': 0.6,  # Very heavy dropout (60%)
    'use_mixup': True,
    'mixup_alpha': 0.8,  # Very strong mixup
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
