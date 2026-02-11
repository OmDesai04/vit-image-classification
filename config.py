DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'crop_size': 150,  # AGGRESSIVE corner cropping - removes 33% from all edges!
}

MODEL_CONFIG = {
    'model_name': 'vit_tiny_patch16_224',  # ViT-Tiny (5.7M params) - 15x smaller to prevent overfitting
    'pretrained': True,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 5e-5,  # Even lower LR
    'weight_decay': 0.2,  # Maximum weight decay
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
    'label_smoothing': 0.6,  # EXTREME label smoothing (60%)
    'dropout': 0.7,  # EXTREME dropout (70%)
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
