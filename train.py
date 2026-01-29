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

        self.criterion = nn.CrossEntropyLoss()

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
        self.early_stopping_patience = 7
        self.early_stopping_counter = 0

        self.history = {
            'train_acc': [],
            'val_acc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }


    def train_epoch(self, epoch):
        self.model.train()
        correct, total = 0, 0

        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return correct / total


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


    def train(self):
        for epoch in range(self.config['epochs']):
            start_time = time.time()

            train_acc = self.train_epoch(epoch)
            val_loss, val_acc, precision, recall, f1, cm = self.validate(epoch)

            self.scheduler.step(val_loss)

            epoch_time = (time.time() - start_time) / 60

            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['precision'].append(precision)
            self.history['recall'].append(recall)
            self.history['f1'].append(f1)

            print(f"\nEpoch {epoch+1}")
            print(f"  Train Accuracy: {train_acc*100:.2f}%")
            print(f"  Val Accuracy:   {val_acc*100:.2f}%")
            print(f"  F1 Score:       {f1:.4f}")
            print(f"  Epoch Time:     {epoch_time:.2f} min")
            print("-" * 60)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("\nEarly stopping triggered.")
                    break

        self.plot_confusion_matrix(cm)

        print("\nFINAL VALIDATION METRICS")
        print(f"Accuracy : {self.history['val_acc'][-1]*100:.2f}%")
        print(f"Precision: {self.history['precision'][-1]:.4f}")
        print(f"Recall   : {self.history['recall'][-1]:.4f}")
        print(f"F1 Score : {self.history['f1'][-1]:.4f}")
        print("=" * 60)


def main():
    config = {
        'data_root': "/content/drive/MyDrive/split_dataset",
        'batch_size': 32,
        'num_workers': 2,
        'image_size': 224,
        'epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'model_name': 'vit_base_patch16_224',
        'pretrained': True,
        'freeze_backbone': False,
        'output_dir': "/content/drive/MyDrive/outputs"
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

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

    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()