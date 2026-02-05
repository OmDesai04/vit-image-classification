import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from dataset_loader import create_dataloaders
from model import create_model


class Trainer:

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Strong label smoothing to prevent 100% accuracy and overconfidence
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.early_stopping_patience = 5
        self.early_stopping_counter = 0

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'epoch_times': []
        }
        
        self.total_training_time = 0


    def train_epoch(self, epoch):
        self.model.train()
        correct, total = 0, 0
        total_loss = 0

        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return correct / total, total_loss / total


    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images, labels = images.to(self.device), labels.to(self.device)
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
                'total_training_time': self.format_time(self.total_training_time)
            },
            'final_metrics': {
                'train_accuracy': float(self.history['train_acc'][-1]),
                'val_accuracy': float(self.history['val_acc'][-1]),
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
        
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            train_acc, train_loss = self.train_epoch(epoch)
            val_loss, val_acc, precision, recall, f1, cm = self.validate(epoch)

            self.scheduler.step(val_loss)

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
            print(f"  Train Accuracy: {train_acc*100:.2f}%")
            print(f"  Val Loss:       {val_loss:.4f}")
            print(f"  Val Accuracy:   {val_acc*100:.2f}%")
            print(f"  Precision:      {precision:.4f}")
            print(f"  Recall:         {recall:.4f}")
            print(f"  F1 Score:       {f1:.4f}")
            print(f"  Epoch Time:     {self.format_time(epoch_time)}")
            print("-" * 60)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
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
        print(f"   Accuracy:  {self.history['val_acc'][-1]*100:.2f}%")
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
    print(f"   Image Size: {config['image_size']}x{config['image_size']}")
    print(f"   Batch Size: {config['batch_size']}")
    print("-" * 80)
    
    # Model Information
    print("\n🤖 MODEL INFORMATION:")
    print(f"   Model Name: {config['model_name']}")
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
    config = {
        'data_root': "/content/drive/MyDrive/split_dataset",
        'batch_size': 32,
        'num_workers': 2,
        'image_size': 224,
        'epochs': 30,
        'learning_rate': 5e-5,
        'weight_decay': 0.1,
        'model_name': 'vit_tiny_patch16_224',  # ViT-Tiny (5.54M params) - Pure ViT
        'pretrained': True,  # Use pretrained ImageNet weights
        'freeze_backbone': False,
        'output_dir': "/content/drive/MyDrive/outputs"
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, _, num_classes, _ = create_dataloaders(
        config['data_root'],
        config['batch_size'],
        config['num_workers'],
        config['image_size']
    )

    model = create_model(
        num_classes,
        config['model_name'],
        config['pretrained'],
        config['freeze_backbone']
    )

    # Print detailed information before training
    print_training_info(config, device, train_loader, val_loader, num_classes, model)

    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()
