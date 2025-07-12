from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from dataPipeline.dataSet import dataPipeline
from model.model import model
import numpy as np
import os
from PIL import Image
import dataPipeline.embeddingUtils as eu
import torch.nn.functional as F
import mediapipe as mp


def load_full_dataset(path, classes=12):
    # complete dataset without train split division 
    files = [d for d in os.listdir(path) if d.isdigit()]
    files = sorted(files, key=lambda x: int(x))
    
    # Load graphs and images
    graphs, normIms = eu.loadGraphs(path)
    
    imageData = []
    vertexData = []
    edgeData = []
    targets = []
    toTensor = transforms.ToTensor()
    
    for pose_idx in range(len(files)):
        if pose_idx < len(normIms) and pose_idx < len(graphs):
            for elem_idx in range(len(normIms[pose_idx])):
                if elem_idx < len(graphs[pose_idx]):
                    # Load image
                    im_path = normIms[pose_idx][elem_idx]
                    im = Image.open(im_path)
                    im = toTensor(im)
                    imageData.append(im)
                    
                    # Load graph data
                    vertexData.append(graphs[pose_idx][elem_idx][0])
                    edgeData.append(graphs[pose_idx][elem_idx][1])
                    
                    # Create one-hot target
                    target = F.one_hot(torch.tensor(pose_idx), num_classes=classes)
                    targets.append(target)
    
    # Stack all data
    imageData = torch.stack(imageData)
    vertexData = torch.stack(vertexData)
    edgeData = torch.stack(edgeData)
    targets = torch.stack(targets)
    
    return imageData, vertexData, edgeData, targets


def stratified_k_fold_evaluation(dataset_path, model_class, device, k=5, batches=16, classes=12, epochs=10):
 
    # Load the full dataset
    imageData, vertexData, edgeData, targets = load_full_dataset(dataset_path, classes)
    
    # Convert one-hot targets to class indices for stratification
    class_indices = targets.argmax(dim=1).numpy()
    
    print(f"Dataset loaded: {len(imageData)} samples")
    print(f"Class distribution: {np.bincount(class_indices)}")
    print(f"Unique classes: {np.unique(class_indices)}")
    
    # Validate that we have valid class indices
    if len(np.unique(class_indices)) < 2:
        raise ValueError(f"Need at least 2 classes for stratification, got {len(np.unique(class_indices))}")
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    # Initialize backbone (ResNet50) once
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    
    # Get the indices for the full dataset
    indices = np.arange(len(imageData))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, class_indices)):
        print(f"\n Fold {fold + 1}/{k} ")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create training data
        train_images = imageData[train_idx]
        train_vertices = vertexData[train_idx]
        train_edges = edgeData[train_idx]
        train_labels = targets[train_idx]
        
        # Create validation data
        val_images = imageData[val_idx]
        val_vertices = vertexData[val_idx]
        val_edges = edgeData[val_idx]
        val_labels = targets[val_idx]
        
        # Create tensor datasets
        train_img_dataset = TensorDataset(train_images, train_labels)
        train_vert_dataset = TensorDataset(train_vertices, train_labels)
        train_edge_dataset = TensorDataset(train_edges, train_labels)
        
        val_img_dataset = TensorDataset(val_images, val_labels)
        val_vert_dataset = TensorDataset(val_vertices, val_labels)
        val_edge_dataset = TensorDataset(val_edges, val_labels)
        
        # Create data loaders
        train_im = DataLoader(train_img_dataset, batch_size=batches, shuffle=True)
        train_ve = DataLoader(train_vert_dataset, batch_size=batches, shuffle=True)
        train_ed = DataLoader(train_edge_dataset, batch_size=batches, shuffle=True)
        
        val_im = DataLoader(val_img_dataset, batch_size=batches, shuffle=False)
        val_ve = DataLoader(val_vert_dataset, batch_size=batches, shuffle=False)
        val_ed = DataLoader(val_edge_dataset, batch_size=batches, shuffle=False)
        
        # Create model with correct parameter names
        model_instance = model_class(
            backbone=feature_extractor, 
            device=device, 
            numLayers=6,  
            num_classes=classes,
            dropout_rate=0.5
        )
        model_instance.to(device)
        
        # Training setup
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_instance.parameters(), lr=1e-5, weight_decay=1e-5)
        
        # Train using the existing trainL method
        train_results, eval_results = model_instance.trainL(
            dataLoaders=[[train_im, train_ve, train_ed], [val_im, val_ve, val_ed]],
            lossFunc=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            patience=5
        )
        
        # Extract evaluation metrics from the last epoch
        fold_metrics = {
            'train_loss': train_results[0][-1],
            'train_acc': train_results[1][-1],
            'val_loss': train_results[2][-1],
            'val_acc': train_results[3][-1],
            'predictions': eval_results[0],
            'targets': eval_results[1]
        }
        model_instance.save(path=f"D:\dionigi\Documents\Python scripts\\aml2025Data\models\\bestVal{fold}_{model_instance.bestVal}.pth")
        
        fold_results.append(fold_metrics)
        
        # Print fold results
        print(f"Fold {fold + 1} Results:")
        print(f"  Train Loss: {fold_metrics['train_loss']:.4f}")
        print(f"  Train Acc: {fold_metrics['train_acc']:.4f}")
        print(f"  Val Loss: {fold_metrics['val_loss']:.4f}")
        print(f"  Val Acc: {fold_metrics['val_acc']:.4f}")
    
    # Aggregate results across folds
    aggregated_metrics = {
        'mean_train_loss': np.mean([f['train_loss'] for f in fold_results]),
        'std_train_loss': np.std([f['train_loss'] for f in fold_results]),
        'mean_train_acc': np.mean([f['train_acc'] for f in fold_results]),
        'std_train_acc': np.std([f['train_acc'] for f in fold_results]),
        'mean_val_loss': np.mean([f['val_loss'] for f in fold_results]),
        'std_val_loss': np.std([f['val_loss'] for f in fold_results]),
        'mean_val_acc': np.mean([f['val_acc'] for f in fold_results]),
        'std_val_acc': np.std([f['val_acc'] for f in fold_results])
    }
    
    return aggregated_metrics, fold_results


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset path
    dataset_path = "D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm"  
    
    # Cross-validation parameters
    k_folds = 3
    batch_size = 16
    num_classes = 9
    epochs_per_fold = 10

    try:
        # Run stratified cross-validation
        aggregated_results, fold_results = stratified_k_fold_evaluation(
            dataset_path=dataset_path,
            model_class=model,
            device=device,
            k=k_folds,
            batches=batch_size,
            classes=num_classes,
            epochs=epochs_per_fold
        )

        # Print aggregated results
        print("\nCross-Validation Results")
        print(f"Average Training Loss: {aggregated_results['mean_train_loss']:.4f} ± {aggregated_results['std_train_loss']:.4f}")
        print(f"Average Training Accuracy: {aggregated_results['mean_train_acc']:.4f} ± {aggregated_results['std_train_acc']:.4f}")
        print(f"Average Validation Loss: {aggregated_results['mean_val_loss']:.4f} ± {aggregated_results['std_val_loss']:.4f}")
        print(f"Average Validation Accuracy: {aggregated_results['mean_val_acc']:.4f} ± {aggregated_results['std_val_acc']:.4f}")
        
        # Print individual fold results
        print("\nIndividual Fold Results")
        for i, fold_result in enumerate(fold_results):
            print(f"Fold {i+1}: Train Acc: {fold_result['train_acc']:.4f}, Val Acc: {fold_result['val_acc']:.4f}")
            
        return aggregated_results, fold_results
        
    except Exception as e:
        print(f"Error during cross-validation: {str(e)}")
        return None, None


if __name__ == "__main__":   
    results = main()