import torch
from args import get_args
from dataset import load_data
from model import initialize_model
from trainer import train_model
from evaluate import evaluate_model
from utils import plot_metrics, save_results, plot_roc_curve, plot_confusion_matrix

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
    model = initialize_model(model_name=args.model_name).to(device)

    # Train the model
    print("Starting training...")
    fold_accuracies, fold_val_losses, all_train_losses, all_val_losses, all_val_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader, 
    device=device,
    args=args
)


    # Save training metrics
    if args.k_folds > 1:
        save_results(fold_accuracies, fold_val_losses, args.k_folds, args.results_csv)

    # Plot metrics for the last fold or overall (if applicable)
    plot_metrics(
        all_train_losses, all_val_losses, all_val_accuracies,
        output_file=f"plots/{args.model_name}_training_metrics.png"
    )

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    evaluation_results = evaluate_model(model, test_loader, device)

    # Extract predictions and labels for plotting additional metrics
    test_predictions = evaluation_results["predictions"]
    test_labels = evaluation_results["labels"]

    # Plot and save the confusion matrix and ROC curve
    print("Plotting the confusion matrix and ROC curve...")
    plot_confusion_matrix(
        test_labels, test_predictions, class_names=["Normal", "Pneumonia"],
        output_file=f"plots/{args.model_name}_confusion_matrix.png"
    )
    plot_roc_curve(
        test_labels, test_predictions,
        output_file=f"plots/{args.model_name}_roc_curve.png"
    )

    print("\nEvaluation Metrics:")
    for key, value in evaluation_results["metrics"].items():
        print(f"{key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()

