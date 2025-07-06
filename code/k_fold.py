from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from dataPipeline.dataSet import dataPipeline
from model import model
import numpy as np


def stratified_k_fold_evaluation(dataset_path, model_class, device, k=5, batches=16, classes=12, epochs=10):

    # Load the full dataset without splitting
    full_dataset = dataPipeline(dataset_path, split=None, batches=batches, classes=classes)
    
    # Extract labels and convert from one-hot to class indices for stratification
    labels = full_dataset.tensors[-1]  # Get the one-hot encoded labels
    
    # Debug: Check the shape and type of labels
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"First few labels: {labels[:3]}")
    
    # Handle different label formats
    if labels.dim() == 3 and labels.shape[1] == 1:
        # Shape is (N, 1, num_classes) - squeeze the middle dimension
        labels = labels.squeeze(1)
    elif labels.dim() == 1:
        # Already class indices
        class_indices = labels.numpy()
    
    # Convert from one-hot to class indices
    if labels.dim() == 2 and labels.shape[1] > 1:
        # One-hot encoded labels
        class_indices = labels.argmax(dim=1).numpy()
    else:
        # Handle other formats
        class_indices = labels.squeeze().numpy()
    
    # Ensure class_indices are integers
    class_indices = class_indices.astype(int)
    
    print(f"Class indices shape: {class_indices.shape}")
    print(f"Class indices dtype: {class_indices.dtype}")
    print(f"First few class indices: {class_indices[:5]}")
    print(f"Unique classes: {np.unique(class_indices)}")
    
    # Validate that we have valid class indices
    if len(np.unique(class_indices)) < 2:
        raise ValueError(f"Need at least 2 classes for stratification, got {len(np.unique(class_indices))}")
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    # Initialize backbone (ResNet50) once
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    
    # Get the indices for the full dataset
    indices = np.arange(len(full_dataset))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, class_indices)):
        print(f"\n=== Fold {fold + 1}/{k} ===")
        
        # Create subsets for each data type
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # Create separate data loaders for images, vertices, and edges
        # We need to extract each component separately
        train_images = torch.stack([full_dataset[i][0] for i in train_idx])
        train_vertices = torch.stack([full_dataset[i][1] for i in train_idx])
        train_edges = torch.stack([full_dataset[i][2] for i in train_idx])
        train_labels = torch.stack([full_dataset[i][3] for i in train_idx])
        
        val_images = torch.stack([full_dataset[i][0] for i in val_idx])
        val_vertices = torch.stack([full_dataset[i][1] for i in val_idx])
        val_edges = torch.stack([full_dataset[i][2] for i in val_idx])
        val_labels = torch.stack([full_dataset[i][3] for i in val_idx])
        
        # Create tensor datasets
        from torch.utils.data import TensorDataset
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
        
        # Create model
        model = model_class(backbone=feature_extractor, device=device, 
                          dimEmbeddings=7080, gnn_dim=768, num_classes=classes)
        model.to(device)
        
        # Training setup
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Train using your existing trainL method
        train_results, eval_results = model.trainL(
            dataLoaders=[[train_im, train_ve, train_ed], [val_im, val_ve, val_ed]],
            lossFunc=loss_fn,
            optimizer=optimizer,
            epochs=epochs
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
        
        fold_results.append(fold_metrics)
    
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
    # Configurazione del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Percorso del dataset
    dataset_path = "/Users/valentinazingarello/Downloads/dataNormL"  # Sostituisci con il tuo percorso
    
    # Parametri per la cross-validazione
    k_folds = 5
    batch_size = 16
    num_classes = 12
    epochs_per_fold = 10  # Numero di epoche per ogni fold

    # Esegui la cross-validazione stratificata
    aggregated_results, fold_results = stratified_k_fold_evaluation(
        dataset_path=dataset_path,
        model_class=model,
        device=device,
        k=k_folds,
        batches=batch_size,
        classes=num_classes,
        epochs=epochs_per_fold
    )

    # Stampa i risultati aggregati
    print("\n=== Cross-Validation Results ===")
    print(f"Average Training Loss: {aggregated_results['mean_train_loss']:.4f} ± {aggregated_results['std_train_loss']:.4f}")
    print(f"Average Training Accuracy: {aggregated_results['mean_train_acc']:.4f} ± {aggregated_results['std_train_acc']:.4f}")
    print(f"Average Validation Loss: {aggregated_results['mean_val_loss']:.4f} ± {aggregated_results['std_val_loss']:.4f}")
    print(f"Average Validation Accuracy: {aggregated_results['mean_val_acc']:.4f} ± {aggregated_results['std_val_acc']:.4f}")

if __name__ == "__main__":   
    print(main())