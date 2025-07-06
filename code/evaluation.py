import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from itertools import product
import json
import os
from model import resnet50, ResNet50_Weights
from torch import nn
from dataPipeline.dataSet import DataLoader, Subset


def calculate_accuracy(predictions, targets):
    #Calculate accuracy from predictions and targets
    if isinstance(predictions, list):
        predictions = torch.cat(predictions, dim=0)
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=0)
    
    pred_classes = predictions.argmax(dim=1)
    return accuracy_score(targets.cpu().numpy(), pred_classes.cpu().numpy())

def calculate_top_k_accuracy(predictions, targets, k=3):
    #Calculate top-k accuracy
    if isinstance(predictions, list):
        predictions = torch.cat(predictions, dim=0)
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=0)
    
    return top_k_accuracy_score(
        targets.cpu().numpy(), 
        predictions.cpu().numpy(), 
        k=k, 
        labels=range(predictions.shape[1])
    )

def calculate_per_class_metrics(predictions, targets, num_classes=12):
    #Calculate precision, recall, F1-score per class
    if isinstance(predictions, list):
        predictions = torch.cat(predictions, dim=0)
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=0)
    
    pred_classes = predictions.argmax(dim=1)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        targets.cpu().numpy(), 
        pred_classes.cpu().numpy(),
        labels=range(num_classes),
        average=None,
        zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

def plot_confusion_matrix(predictions, targets, class_names=None, save_path=None):
    #Plot and return confusion matrix
    if isinstance(predictions, list):
        predictions = torch.cat(predictions, dim=0)
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=0)
    
    pred_classes = predictions.argmax(dim=1)
    cm = confusion_matrix(targets.cpu().numpy(), pred_classes.cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    if class_names is None:
        class_names = [f'Pose {i}' for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return cm

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=None):
    #Plot training and validation curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def comprehensive_evaluation(predictions, targets, class_names=None, save_dir=None):
    results = {}
    
    # Basic accuracy
    results['accuracy'] = calculate_accuracy(predictions, targets)
    
    # Top-k accuracy
    results['top_3_accuracy'] = calculate_top_k_accuracy(predictions, targets, k=3)
    results['top_5_accuracy'] = calculate_top_k_accuracy(predictions, targets, k=5)
    
    # Per-class metrics
    per_class = calculate_per_class_metrics(predictions, targets)
    results['per_class_precision'] = per_class['precision']
    results['per_class_recall'] = per_class['recall']
    results['per_class_f1'] = per_class['f1']
    results['per_class_support'] = per_class['support']
    
    # Macro and weighted averages
    results['macro_precision'] = np.mean(per_class['precision'])
    results['macro_recall'] = np.mean(per_class['recall'])
    results['macro_f1'] = np.mean(per_class['f1'])
    
    results['weighted_precision'] = np.average(per_class['precision'], weights=per_class['support'])
    results['weighted_recall'] = np.average(per_class['recall'], weights=per_class['support'])
    results['weighted_f1'] = np.average(per_class['f1'], weights=per_class['support'])
    
    # Confusion matrix
    if save_dir:
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        results['confusion_matrix'] = plot_confusion_matrix(predictions, targets, class_names, cm_path)
    else:
        results['confusion_matrix'] = plot_confusion_matrix(predictions, targets, class_names)
    
    return results

def save_results(results, save_path):
    #Save evaluation results to file
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_results(load_path):
    """Load evaluation results from file"""
    with open(load_path, 'r') as f:
        return json.load(f)

def print_evaluation_summary(results):
    #Print a summary of evaluation results
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
    
    print(f"\nMacro Averages:")
    print(f"  Precision: {results['macro_precision']:.4f}")
    print(f"  Recall: {results['macro_recall']:.4f}")
    print(f"  F1-Score: {results['macro_f1']:.4f}")
    
    print(f"\nWeighted Averages:")
    print(f"  Precision: {results['weighted_precision']:.4f}")
    print(f"  Recall: {results['weighted_recall']:.4f}")
    print(f"  F1-Score: {results['weighted_f1']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for i, f1 in enumerate(results['per_class_f1']):
        print(f"  Class {i}: {f1:.4f}")

def lossAndAccGraph(trainLoss, trainAcc, testLoss, testAcc):
    plot_training_curves(trainLoss, trainAcc, testLoss, testAcc)

def confusionMatAndFScores(targets, predictions):
    results = comprehensive_evaluation(predictions, targets)
    print_evaluation_summary(results)
    return results





