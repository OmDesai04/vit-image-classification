import os
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_loader import create_dataloaders
from model import create_model


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Weight decay: {config['weight_decay']}")
        print(f"Scheduler: {config['scheduler']}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]  ")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self):
        print("\nStarting training...\n")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            train_loss, train_accuracy = self.train_epoch(epoch)
            
            val_loss, val_accuracy = self.validate(epoch)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rates'].append(current_lr)
            
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy*100:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, val_accuracy, is_best=True)
                print(f"  ✓ New best model saved! (Val Acc: {val_accuracy*100:.2f}%)")
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_accuracy*100:.2f}% (Epoch {self.best_epoch+1})")
        
        self.save_checkpoint('final_model.pth', self.config['epochs']-1, val_accuracy, is_best=False)
        
        self.save_history()
        
        self.plot_training_curves()
    
    def save_checkpoint(self, filename, epoch, val_accuracy, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'config': self.config,
            'history': self.history
        }
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"\nTraining history saved to {history_path}")
    
    def plot_training_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        train_acc = [acc * 100 for acc in self.history['train_accuracy']]
        val_acc = [acc * 100 for acc in self.history['val_accuracy']]
        axes[1].plot(epochs, train_acc, 'b-', label='Training Accuracy')
        axes[1].plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        
        plt.close()


def main():
    config = {
        'data_root': 'split_dataset',
        'batch_size': 32,
        'num_workers': 4,
        'image_size': 224,
        'epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'scheduler': 'plateau',
        'model_name': 'vit_base_patch16_224',
        'pretrained': True,
        'freeze_backbone': False,
        'output_dir': 'outputs'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    print("\nCreating model...")
    model = create_model(
        num_classes=num_classes,
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        freeze_backbone=config['freeze_backbone']
    )
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=4)
    
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best model saved at: {output_dir / 'best_model.pth'}")
    print(f"Final model saved at: {output_dir / 'final_model.pth'}")
    print(f"Training curves saved at: {output_dir / 'training_curves.png'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
