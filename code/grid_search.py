from itertools import product
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from dataPipeline.dataSet import dataPipeline
from model import model,weightDataSet  
import numpy as np


def grid_search_hyperparameters(model_class, dataset_path, device, param_grid, epochs=100, batches=16, classes=9):
   
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    results = []
    best_score = -np.inf
    best_train = -np.inf
    best_params = None
    
    # Load dataset once
    train_data, val_data = dataPipeline(dataset_path, split=0.8, batches=batches, classes=classes)
    weightsData = weightDataSet([train_data,val_data])
    
    # Initialize backbone (ResNet50) once
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\nTesting parameters {i+1}/{len(param_combinations)}: {param_dict}")
        
        try:
            # Separate model parameters from optimizer parameters
            model_params = {k: v for k, v in param_dict.items() 
                          if k not in ['lr', 'weight_decay']}
            
            # Create model with only model parameters
            model_instance = model_class(
                backbone=feature_extractor,
                device=device,
                num_classes=classes,
                **model_params
            )
            model_instance.to(device)
            
            # Training setup
            loss_fn = torch.nn.CrossEntropyLoss(weight=weightsData.to(device))
            optimizer = torch.optim.AdamW(model_instance.parameters(), 
                                         lr=param_dict.get('lr', 1e-4), 
                                         weight_decay=param_dict.get('weight_decay', 1e-5))
            
            # Train model
            train_results, eval_results = model_instance.trainL(
                dataLoaders=[train_data, val_data],
                lossFunc=loss_fn,
                optimizer=optimizer,
                epochs=epochs,
                patience=3
            )
            
            # Get final validation accuracy
            val_accuracy = model_instance.bestVal  #train_results[3][-1]  # Last validation accuracy
            train_accuracy = train_results[1][-1]
            
            # Store results
            result = {
                'params': param_dict,
                'val_accuracy': val_accuracy,
                'final_val_loss': train_results[2][-1],
                'final_train_loss': train_results[0][-1],
                'final_train_accuracy': train_results[1][-1],
                'all_train_losses': train_results[0],
                'all_train_accuracies': train_results[1],
                'all_val_losses': train_results[2],
                'all_val_accuracies': train_results[3]
            }
            
            results.append(result)
            
            # Update best parameters
            if val_accuracy > best_score:
                best_score = val_accuracy
                best_params = param_dict
                best_train = train_accuracy 

                
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Train accuracy: {train_accuracy:.4f}")
            
            
        except Exception as e:
            print(f"Error with parameters {param_dict}: {str(e)}")
            continue
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results,
        'best_train' : best_train
    }


def main():
    # Define parameter grid
    param_grid = {
        'lr': [1e-4, 1e-5],                    # optimizer parameter
        'weight_decay': [1e-4, 1e-5],          # optimizer parameter
        'dropout_rate': [0.1, 0.3, 0.5],       # model parameter
        'numLayers': [2, 4, 6]                # model parameter (BERT layers)
    }
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run grid search
    search_results = grid_search_hyperparameters(
        model_class=model,
        dataset_path="D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm",
        device=device,
        param_grid=param_grid,
        epochs=15,
        batches=16,
        classes=9
    )

    # Print best results
    print("\nBest parameters:")
    print(search_results['best_params'])
    print(f"Best validation accuracy: {search_results['best_score']:.4f}")
    print(f"Best train accuracy: {search_results['best_train']:.4f}")

if __name__ == "__main__":
    main()

