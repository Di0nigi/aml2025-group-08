import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import os
from datetime import datetime
import numpy as np

# Import your existing modules
from model import model
from dataPipeline.dataSet import dataPipeline
import evaluation as ev
from torchvision.models import resnet50, ResNet50_Weights


def create_full_dataset(dataset_path, classes=12):
    #Create full dataset without train/test split for cross-validation
    from dataPipeline.dataSet import loadData
    from dataPipeline import embeddingUtils as eu
    from torchvision import transforms
    import torch.nn.functional as F
    
    files = loadData(dataset_path, norm=True)[::2]
    graphs, normIms = eu.loadGraphs(dataset_path)
    
    imageData = []
    vertexData = []
    edgeData = []
    targets = []
    toTensor = transforms.ToTensor()
    
    for pose in range(len(files)):
        for elem in range(len(normIms[pose])):
            im = normIms[pose][elem]
            im = transforms.Image.open(im) if isinstance(im, str) else im
            im = toTensor(im)
            
            imageData.append(im)
            vertexData.append(graphs[pose][elem][0])
            edgeData.append(graphs[pose][elem][1])
            
            target = torch.zeros(classes)
            target[pose] = 1.0
            targets.append(target)
    
    imageData = torch.stack(imageData)
    vertexData = torch.stack(vertexData)
    edgeData = torch.stack(edgeData)
    labels = torch.stack(targets)
    
    return TensorDataset(imageData, vertexData, edgeData, labels)


def run_comprehensive_evaluation(dataset_path, save_dir=None):
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if save_dir is None:
        save_dir = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = create_full_dataset(dataset_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Class names (you can customize these)
    class_names = [f"Pose_{i}" for i in range(12)]
    
    # Standard train/test evaluation
    print("\n1. Standard Train/Test Evaluation")
    print("="*50)
    
    # Create train/test split
    train_data, test_data = dataPipeline(dataset_path, split=0.8, batches=16, classes=12)
    
    # Initialize model
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    featureExtractor = nn.Sequential(*list(resnet.children())[:-1])
    
    test_model = model(backbone=featureExtractor, device=device)
    test_model.to(device)
    
    # Train model
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    [trainLoss, trainAcc, valLoss, valAcc], [predictions, targets] = test_model.trainL(
        dataLoaders=train_data, 
        lossFunc=lossFunction, 
        optimizer=optimizer, 
        epochs=15
    )
    
    # Plot training curves
    ev.plot_training_curves(trainLoss, trainAcc, valLoss, valAcc, 
                           save_path=os.path.join(save_dir, "training_curves.png"))
    
    # Comprehensive evaluation
    print("\nEvaluating model performance...")
    results = ev.comprehensive_evaluation(predictions, targets, class_names, save_dir)
    ev.print_evaluation_summary(results)
    
    # Save results
    ev.save_results(results, os.path.join(save_dir, "evaluation_results.json"))
    
    # Cross-validation evaluation
    print("\n2. Stratified K-Fold Cross-Validation")
    print("="*50)
    
    try:
        # Note: This is a simplified version. For full implementation, 
        # you'd need to adapt the stratified_k_fold_evaluation function
        # to work with your specific model architecture
        
        print("Performing 5-fold cross-validation...")
        # cv_results, fold_results = ev.stratified_k_fold_evaluation(
        #     model, dataset, device, k=5,
        #     backbone=featureExtractor
        # )
        # print("Cross-validation results:")
        # for metric, value in cv_results.items():
        #     print(f"  {metric}: {value:.4f}")
        
        print("Cross-validation implementation requires adaptation to your specific data pipeline.")
        print("See the evaluationUtils.py file for the framework.")
        
    except Exception as e:
        print(f"Cross-validation error: {str(e)}")
    
    # 3. Hyperparameter tuning
    print("\n3. Hyperparameter Grid Search")
    print("="*50)
    
    # Define hyperparameter grid
    param_grid = {
        'gnn_dim': [512, 768, 1024],
        'dropout_rate': [0.1, 0.2, 0.3],
        'num_layers': [6, 8, 12]
    }
    
    try:
        print("Performing grid search...")
        print("Parameter grid:", param_grid)
        
        # grid_results = ev.grid_search_hyperparameters(
        #     model, dataset, device, param_grid, cv_folds=3
        # )
        # 
        # print(f"Best parameters: {grid_results['best_params']}")
        # print(f"Best CV score: {grid_results['best_score']:.4f}")
        
        print("Grid search implementation requires adaptation to your specific model.")
        print("See the evaluationUtils.py file for the framework.")
        
    except Exception as e:
        print(f"Grid search error: {str(e)}")
    
    # 4. Additional analyses
    print("\n4. Additional Analysis")
    print("="*50)
    
    # Per-class analysis
    print("\nPer-class performance analysis:")
    for i, (precision, recall, f1, support) in enumerate(zip(
        results['per_class_precision'], 
        results['per_class_recall'], 
        results['per_class_f1'], 
        results['per_class_support']
    )):
        print(f"Class {i} ({class_names[i]}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Support: {support}")
    
    # Error analysis
    print(f"\nConfusion Matrix Analysis:")
    cm = results['confusion_matrix']
    print(f"Most confused classes:")
    
    # Find most confused pairs (excluding diagonal)
    np.fill_diagonal(cm, 0)
    confused_pairs = np.unravel_index(np.argsort(cm.ravel())[-5:], cm.shape)
    for actual, predicted in zip(confused_pairs[0][::-1], confused_pairs[1][::-1]):
        if cm[actual, predicted] > 0:
            print(f"  {class_names[actual]} -> {class_names[predicted]}: {cm[actual, predicted]} instances")
    
    print(f"\nResults saved to: {save_dir}")
    return results, save_dir


def run_quick_evaluation(dataset_path):
    #limited quick evaluation for testing purposes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data with your existing pipeline
    data = dataPipeline(dataset_path, split=0.8, batches=16, classes=12)
    
    # Initialize model
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    featureExtractor = nn.Sequential(*list(resnet.children())[:-1])
    
    test_model = model(backbone=featureExtractor, device=device)
    test_model.to(device)
    
    # Quick training
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    [trainLoss, trainAcc, valLoss, valAcc], [predictions, targets] = test_model.trainL(
        dataLoaders=data, 
        lossFunc=lossFunction, 
        optimizer=optimizer, 
        epochs=3
    )
    
    # Quick evaluation
    results = ev.comprehensive_evaluation(predictions, targets)
    ev.print_evaluation_summary(results)
    
    return results


def main():
    
    dataset_path = "/Users/valentinazingarello/Downloads/dataNormL" 
    
    print("Yoga Pose Classification - Comprehensive Evaluation")
    print("="*60)
    
    # Choose evaluation type
    evaluation_type = input("Choose evaluation type:\n1. Quick evaluation\n2. Comprehensive evaluation\nEnter choice (1 or 2): ")
    
    if evaluation_type == "1":
        print("\nquick evaluation")
        results = run_quick_evaluation(dataset_path)
    elif evaluation_type == "2":
        print("\ncomprehensive evaluation")
        results, save_dir = run_comprehensive_evaluation(dataset_path)
        print(f"\nDetailed results saved in: {save_dir}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()