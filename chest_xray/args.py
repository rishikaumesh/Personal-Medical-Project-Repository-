import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Model Arguments")

    # Dataset paths
    parser.add_argument('--train_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/train', help='Path to the training dataset')
    parser.add_argument('--val_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/val', help='Path to the validation dataset')
    parser.add_argument('--test_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/test', help='Path to the test dataset')
    parser.add_argument('--augmented_train_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/augmented/train', help='Path to the augmented training dataset')
    parser.add_argument('--augmented_val_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/augmented/val', help='Path to the augmented validation dataset')
    parser.add_argument('--augmented_test_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/augmented/test', help='Path to the augmented test dataset')
    
    # Output directories
    parser.add_argument('--output_dir', type=str, default='/home/user/persistent/chest_xray/chest_xray/augmented', help='Base output directory for augmented images')
    
    # Image properties
    parser.add_argument('--img_size', type=int, default=224, help='Image size (square dimensions)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--scheduler_patience', type=int, default=2, help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.3, help='Factor for learning rate scheduler')
    
    # Cross-validation
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for K-Fold Cross-Validation')
    
    # Data augmentation and sampling
    parser.add_argument('--use_data_augmentation', action='store_true', help='Enable data augmentation for training')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='Enable weighted sampler for handling class imbalance')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping during training')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu)')

    # Results
    parser.add_argument('--results_csv', type=str, default='results.csv', help='CSV file to save training results')

    args = parser.parse_args()
    return args
