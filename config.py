DATA_CONFIG = {
    'data_root': 'split_dataset',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4,
}

MODEL_CONFIG = {
    'model_name': 'vit_base_patch16_224',  # ViT-Base (85.85M params) - Pure ViT with patch16
    'pretrained': True,  # Use pretrained weights
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
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
