import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from dataset_loader import create_dataloaders
from model import create_model
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG


class Trainer:

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Label smoothing for better generalization
        label_smoothing = config.get('label_smoothing', 0.1)
        class_weights = None
        if config.get('use_class_weights', False):
            labels = np.array(self.train_loader.dataset.labels)
            class_counts = np.bincount(labels)
            class_counts = np.maximum(class_counts, 1)
            inv_freq = 1.0 / class_counts
            normalized = inv_freq / inv_freq.mean()
            class_weights = torch.tensor(normalized, dtype=torch.float32, device=device)
            print("Using class-weighted CrossEntropyLoss")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # OneCycleLR for faster convergence and better accuracy
        total_steps = len(train_loader) * config['epochs']
        if config.get('scheduler') == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.get('max_lr', 3e-3),
                total_steps=total_steps,
                pct_start=config.get('warmup_epochs', 3) / config['epochs'],
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )
            self.scheduler_step_per_batch = True
        else:
            # Fallback to ReduceLROnPlateau
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            self.scheduler_step_per_batch = False

        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0

        # Mixed precision training for 2-3x speedup
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Gradient clipping to prevent instability
        self.gradient_clip = config.get('gradient_clip', 1.0)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        self.total_training_time = 0
        
        # Mixup for better accuracy and generalization
        self.use_mixup = config.get('use_mixup', True)
        self.mixup_alpha = config.get('mixup_alpha', 0.2)

    @staticmethod
    def _reported_accuracy(acc):
        """Return raw accuracy value for honest reporting."""
        return float(acc)
    
    def mixup_data(self, x, y, alpha=0.4):
        """Apply mixup augmentation to prevent overfitting"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Compute mixup loss"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


    def train_epoch(self, epoch):
        self.model.train()
        correct, total = 0, 0
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training
            with autocast('cuda', enabled=self.use_amp):
                if self.use_mixup:
                    # Apply mixup augmentation
                    mixed_images, y_a, y_b, lam = self.mixup_data(images, labels, self.mixup_alpha)
                    outputs = self.model(mixed_images)
                    loss = self.mixup_criterion(outputs, y_a, y_b, lam)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Step scheduler per batch if using OneCycleLR (after optimizer.step)
            if self.scheduler_step_per_batch:
                self.scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        return correct / total, total_loss / total


    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # Mixed precision for validation too
                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average='macro')

        cm = confusion_matrix(all_labels, all_preds)

        return val_loss, acc, precision, recall, f1, cm


    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300)
        plt.close()


    # ----------- ADDED (DO NOT CHANGE EXISTING) -----------
    def plot_normalized_confusion_matrix(self, cm):
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Normalized)")
        plt.savefig(self.output_dir / "confusion_matrix_normalized.png", dpi=300)
        plt.close()
    # -----------------------------------------------------


    def plot_accuracy_graph(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['train_acc']) + 1)
        plt.plot(epochs, [acc * 100 for acc in self.history['train_acc']], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, [acc * 100 for acc in self.history['val_acc']], 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_graph.png", dpi=300)
        plt.close()


    def plot_loss_graph(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_graph.png", dpi=300)
        plt.close()


    def plot_metrics_graph(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['precision']) + 1)
        plt.plot(epochs, self.history['precision'], 'g-', label='Precision', linewidth=2)
        plt.plot(epochs, self.history['recall'], 'b-', label='Recall', linewidth=2)
        plt.plot(epochs, self.history['f1'], 'r-', label='F1 Score', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision, Recall, and F1 Score', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_graph.png", dpi=300)
        plt.close()


    def save_metrics_to_file(self):
        import json
        
        metrics = {
            'training_summary': {
                'total_epochs': len(self.history['train_acc']),
                'best_val_loss': float(self.best_val_loss),
                'best_val_acc': float(self.best_val_acc),
                'best_val_acc_reported': float(self._reported_accuracy(self.best_val_acc)),
                'total_training_time': self.format_time(self.total_training_time)
            },
            'final_metrics': {
                'train_accuracy': float(self.history['train_acc'][-1]),
                'val_accuracy': float(self.history['val_acc'][-1]),
                'train_accuracy_reported': float(self._reported_accuracy(self.history['train_acc'][-1])),
                'val_accuracy_reported': float(self._reported_accuracy(self.history['val_acc'][-1])),
                'precision': float(self.history['precision'][-1]),
                'recall': float(self.history['recall'][-1]),
                'f1_score': float(self.history['f1'][-1])
            },
            'per_epoch_metrics': {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'train_accuracy': [float(x) for x in self.history['train_acc']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history['val_acc']],
                'precision': [float(x) for x in self.history['precision']],
                'recall': [float(x) for x in self.history['recall']],
                'f1_score': [float(x) for x in self.history['f1']],
                'learning_rates': [float(x) for x in self.history['learning_rates']],
                'epoch_times': [self.format_time(t) for t in self.history['epoch_times']]
            },
            'model_config': self.config
        }
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✅ All metrics saved to: {self.output_dir / 'training_metrics.json'}")


    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes} min {secs} sec"


    def train(self):
        training_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting Training with:")
        print(f"   - Mixed Precision (AMP): {'✓' if self.use_amp else '✗'}")
        print(f"   - Mixup Augmentation: {'✓' if self.use_mixup else '✗'}")
        if self.use_mixup:
            print("   - Note: Train accuracy under MixUp is a proxy and may appear lower than validation accuracy")
        print(f"   - Class Weighted Loss: {'✓' if self.config.get('use_class_weights', False) else '✗'}")
        print(f"   - Gradient Clipping: {self.gradient_clip}")
        print(f"   - Scheduler: {'OneCycleLR' if self.scheduler_step_per_batch else 'ReduceLROnPlateau'}")
        print(f"{'='*80}\n")
        
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            train_acc, train_loss = self.train_epoch(epoch)
            val_loss, val_acc, precision, recall, f1, cm = self.validate(epoch)

            # Step scheduler per epoch if not OneCycleLR
            if not self.scheduler_step_per_batch:
                self.scheduler.step(val_loss)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            epoch_time = time.time() - epoch_start_time

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['precision'].append(precision)
            self.history['recall'].append(recall)
            self.history['f1'].append(f1)
            self.history['epoch_times'].append(epoch_time)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss:     {train_loss:.4f}")
            print(f"  Train Accuracy: {self._reported_accuracy(train_acc)*100:.2f}%")
            print(f"  Val Loss:       {val_loss:.4f}")
            print(f"  Val Accuracy:   {self._reported_accuracy(val_acc)*100:.2f}%")
            print(f"  Precision:      {precision:.4f}")
            print(f"  Recall:         {recall:.4f}")
            print(f"  F1 Score:       {f1:.4f}")
            print(f"  Learning Rate:  {current_lr:.2e}")
            print(f"  Epoch Time:     {self.format_time(epoch_time)}")
            print("-" * 60)

            # Save best model based on validation accuracy (better metric than loss)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_name': self.config.get('model_name', ''),
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, self.output_dir / "best_model.pth")
                print("  ✅ Model saved!")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("\n⚠️  Early stopping triggered.")
                    break

        self.total_training_time = time.time() - training_start_time
        
        print("\n" + "="*80)
        print(" " * 30 + "TRAINING COMPLETED")
        print("="*80)
        print(f"\n⏱️  Total Training Time: {self.format_time(self.total_training_time)}")
        print(f"📊 Total Epochs: {len(self.history['train_acc'])}")
        
        print("\n📈 FINAL VALIDATION METRICS:")
        print(f"   Accuracy:  {self._reported_accuracy(self.history['val_acc'][-1])*100:.2f}%")
        print(f"   Precision: {self.history['precision'][-1]:.4f}")
        print(f"   Recall:    {self.history['recall'][-1]:.4f}")
        print(f"   F1 Score:  {self.history['f1'][-1]:.4f}")
        print("="*80)
        
        # Save all visualizations and metrics
        print("\n💾 Saving results...")
        self.plot_confusion_matrix(cm)
        self.plot_normalized_confusion_matrix(cm)
        self.plot_accuracy_graph()
        self.plot_loss_graph()
        self.plot_metrics_graph()
        self.save_metrics_to_file()
        
        print("\n✅ All graphs and metrics saved successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*80 + "\n")


def print_training_info(config, device, train_loader, val_loader, num_classes, model):
    """Print detailed information about model, dataset, and training configuration."""
    print("\n" + "="*80)
    print(" " * 25 + "TRAINING CONFIGURATION")
    print("="*80)
    
    # Device Information
    print("\n📱 DEVICE INFORMATION:")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("-" * 80)
    
    # Dataset Information
    print("\n📊 DATASET INFORMATION:")
    print(f"   Number of Classes: {num_classes}")
    print(f"   Training Samples: {len(train_loader.dataset)}")
    print(f"   Validation Samples: {len(val_loader.dataset)}")
    print(f"   Training Batches: {len(train_loader)}")
    print(f"   Validation Batches: {len(val_loader)}")
    
    # Image preprocessing information
    crop_size = config.get('crop_size', None)
    if crop_size is not None and crop_size > 0:
        print(f"   ✂️  Image Cropping: ENABLED (Center crop to {crop_size}x{crop_size})")
        print(f"   📐 Final Image Size: {config['image_size']}x{config['image_size']}")
        print(f"   ℹ️  Note: Images are center-cropped BEFORE resizing")
    else:
        print(f"   Image Size: {config['image_size']}x{config['image_size']}")
    
    print(f"   Batch Size: {config['batch_size']}")
    print("-" * 80)
    
    # Model Information
    print("\n🤖 MODEL INFORMATION:")
    print(f"   Model Name: {config['model_name']}")
    if 'swin' in config['model_name'].lower():
        print("   Architecture: Swin Transformer")
    else:
        print("   Architecture: Vision Transformer")
    print(f"   Pretrained: {config['pretrained']}")
    print(f"   Freeze Backbone: {config['freeze_backbone']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
    print("-" * 80)
    
    # Training Configuration
    print("\n⚙️  TRAINING CONFIGURATION:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Weight Decay: {config['weight_decay']}")
    print(f"   Optimizer: AdamW")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Loss Function: CrossEntropyLoss")
    print(f"   Output Directory: {config['output_dir']}")
    print("-" * 80)
    
    print("\n🚀 STARTING TRAINING...")
    print("="*80 + "\n")


def main():
    # Merge all config sections into one
    config = {
        **DATA_CONFIG,
        **MODEL_CONFIG,
        **TRAIN_CONFIG,
        **OUTPUT_CONFIG
    }
    
    # ========================================================================
    # CHANGE YOUR PATHS HERE:
    # ========================================================================
    config['data_root'] = 'split_dataset'  # Change to your dataset path
    config['output_dir'] = 'outputs'       # Change to your output path
    # ========================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"🚀 PyTorch Version: {torch.__version__}")
    print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")

    train_loader, val_loader, _, num_classes, class_names = create_dataloaders(
        config['data_root'],
        config['batch_size'],
        config['num_workers'],
        config['image_size'],
        config.get('crop_size', None),
        config.get('image_extensions', None),
        config.get('check_split_overlap', True),
        config.get('split_overlap_strict', True),
        config.get('pin_memory', True),
        config.get('persistent_workers', True),
        config.get('prefetch_factor', 2)
    )
    
    # Save class names to JSON file for inference
    import json
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Saved class names to {output_dir / 'class_names.json'}")

    model = create_model(
        num_classes,
        config['model_name'],
        config['pretrained'],
        config['freeze_backbone'],
        config.get('dropout', 0.1)
    )
    
    # PyTorch 2.0+ compile for ~30% speedup (if available)
    if config.get('use_compile', False) and hasattr(torch, 'compile'):
        try:
            print("🔥 Compiling model with PyTorch 2.0+ (this may take a minute)...")
            model = torch.compile(model, mode='max-autotune')
            print("✅ Model compiled successfully!\n")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}\n")

    # Print detailed information before training
    print_training_info(config, device, train_loader, val_loader, num_classes, model)

    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()
