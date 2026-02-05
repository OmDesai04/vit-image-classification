import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

from dataset_loader import create_dataloaders
from model import load_model


class ModelEvaluator:
    
    def __init__(self, model, test_loader, class_names, device, output_dir='outputs'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def evaluate(self):
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60 + "\n")
        
        self.model.eval()
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs, 1)
                
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
        
        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_probabilities = np.array(self.all_probabilities)
        
        metrics = self.calculate_metrics()
        
        self.print_metrics(metrics)
        
        self.plot_confusion_matrix()
        
        self.generate_classification_report()
        
        self.save_metrics(metrics)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60 + "\n")
        
        return metrics
    
    def calculate_metrics(self):
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        
        precision = precision_score(self.all_labels, self.all_predictions, average='weighted', zero_division=0)
        recall = recall_score(self.all_labels, self.all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.all_labels, self.all_predictions, average='weighted', zero_division=0)
        
        precision_per_class = precision_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist()
            }
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        print("OVERALL METRICS:")
        print("-" * 60)
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
        print("-" * 60)
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        plt.figure(figsize=(16, 14))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {cm_path}")
        plt.close()
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion'},
            vmin=0,
            vmax=1
        )
        
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_norm_path = self.output_dir / 'confusion_matrix_normalized.png'
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        print(f"Normalized confusion matrix saved to {cm_norm_path}")
        plt.close()
    
    def save_classification_report(self):
        report = classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=self.class_names,
            digits=4
        )
        
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(report)
        
        print(f"Classification report saved to {report_path}")
        
        print(f"\n{report}")
    
    def save_metrics(self, metrics):
        metrics_path = self.output_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {metrics_path}")
    
    def get_predictions_table(self):
        predictions_table = []
        for i, (true_label, pred_label) in enumerate(zip(self.all_labels, self.all_predictions)):
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            correct = true_label == pred_label
            predictions_table.append((i, true_class, pred_class, correct))
        
        return predictions_table
    
    def save_predictions_table(self):
        import csv
        
        predictions_table = self.get_predictions_table()
        
        csv_path = self.output_dir / 'predictions.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample Index', 'True Label', 'Predicted Label', 'Correct'])
            writer.writerows(predictions_table)
        
        print(f"Predictions table saved to {csv_path}")
        
        print("\nSample predictions (first 10):")
        print("-" * 80)
        print(f"{'Index':<10} {'True Label':<15} {'Predicted Label':<15} {'Correct':<10}")
        print("-" * 80)
        for i, (idx, true_label, pred_label, correct) in enumerate(predictions_table[:10]):
            status = "✓" if correct else "✗"
            print(f"{idx:<10} {true_label:<15} {pred_label:<15} {status:<10}")
        print("-" * 80)


def main():
    config = {
        'data_root': 'split_dataset',
        'batch_size': 32,
        'num_workers': 4,
        'image_size': 224,
        'model_path': 'outputs/best_model.pth',
        'model_name': 'vit_base_patch16_224',
        'output_dir': 'outputs'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    class_names_path = Path(config['output_dir']) / 'class_names.json'
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found at {class_names_path}. Please run training first.")
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    
    print("\nLoading test dataset...")
    _, _, test_loader, _, _ = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    print("\nLoading trained model...")
    model = load_model(
        checkpoint_path=config['model_path'],
        num_classes=num_classes,
        model_name=config['model_name'],
        device=device
    )
    
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=config['output_dir']
    )
    
    metrics = evaluator.evaluate()
    
    evaluator.save_predictions_table()
    
    print("\n" + "="*60)
    print("ALL EVALUATION RESULTS SAVED")
    print("="*60)
    print(f"Output directory: {config['output_dir']}")
    print("Files generated:")
    print("  - test_metrics.json (metrics summary)")
    print("  - confusion_matrix.png (visualization)")
    print("  - confusion_matrix_normalized.png (normalized)")
    print("  - classification_report.txt (detailed report)")
    print("  - predictions.csv (all predictions)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
