import torch
from args import get_args
from dataset import load_data
from model import ImprovedPneumoniaDetectionCNN3
from trainer import train_model
from evaluate import evaluate_model
from utils import plot_metrics, save_results

def main():
    # Get arguments
    args = get_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Print dataset paths for debugging
    print(f"Train Dir: {args.train_dir}")
    print(f"Validation Dir: {args.val_dir}")
    print(f"Test Dir: {args.test_dir}")

    # Prepare data loaders
    print("Preparing datasets and dataloaders...")
    train_loader, val_loader, test_loader = load_data(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    # Print DataLoader sizes for debugging
    print(f"Train Loader Length: {len(train_loader)}")
    print(f"Validation Loader Length: {len(val_loader)}")
    print(f"Test Loader Length: {len(test_loader)}")

    # Initialize the model
    print("Initializing the model...")
    model = ImprovedPneumoniaDetectionCNN3().to(device)

    # Train the model
    print("Starting training...")
    fold_accuracies, fold_val_losses, all_train_losses, all_val_losses, all_val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args
    )

    # Save training metrics
    if args.k_folds > 1:
        save_results(fold_accuracies, fold_val_losses, args.k_folds, args.results_csv)

    # Plot metrics for the last fold or overall (if applicable)
    if args.k_folds == 1:
        plot_metrics(all_train_losses, all_val_losses, all_val_accuracies)
    else:
        print("Metrics for each fold saved in CSV file.")

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    test_metrics = evaluate_model(model, test_loader, device)

    print("\nEvaluation Metrics:")
    for key, value in test_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
