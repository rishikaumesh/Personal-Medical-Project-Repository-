import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from utils import plot_metrics, save_results, plot_roc_curve, plot_confusion_matrix
from evaluate import evaluate_model


def train_model(model, train_loader, val_loader, test_loader, device, args):
    """
    Train the Pneumonia Detection CNN model with K-Fold Cross-Validation.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to use for training.
        args (Namespace): Argument parser namespace with hyperparameters.

    Returns:
        tuple: Training metrics including fold accuracies, losses, and validation accuracies.
    """
    print(f"Training model '{args.model_name}' on device: {device}")

    # Hyperparameters
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    k_folds = args.k_folds

    # Define K-Fold cross-validator
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Directories for results and plots
    results_dir = f"results/{args.model_name}"
    plots_dir = f"plots/{args.model_name}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Lists to store metrics across all folds
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    fold_accuracies = []
    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
        print(f"\nStarting Fold {fold + 1}/{k_folds}")

        # Create subsets for the fold
        train_subset = Subset(train_loader.dataset, train_idx)
        val_subset = Subset(train_loader.dataset, val_idx)

        # DataLoader objects for the current fold
        fold_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        fold_val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Reset model parameters
        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        model.to(device)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.3, verbose=True)

        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0

        # Metrics for each fold
        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training phase
            model.train()
            running_loss = 0.0
            for images, labels in fold_train_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(fold_train_loader))
            print(f"Training Loss: {train_losses[-1]:.4f}")

            # Validation phase
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in fold_val_loader:
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    predicted = (outputs > 0.5).int()
                    total += labels.size(0)
                    correct += (predicted == labels.int()).sum().item()

            val_accuracy = 100 * correct / total
            val_losses.append(val_loss / len(fold_val_loader))
            val_accuracies.append(val_accuracy)

            print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Step learning rate scheduler
            scheduler.step(val_accuracy)

            # Save the best model for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{results_dir}/best_model_fold_{fold + 1}.pth")
                print(f"Best model for Fold {fold + 1} saved.")

            # Early stopping
            if args.early_stopping:
                if val_loss < best_val_loss:
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter > args.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        fold_accuracies.append(max(val_accuracies))
        fold_val_losses.append(min(val_losses))
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_val_accuracies.extend(val_accuracies)

    # Calculate average results across folds
    avg_accuracy = sum(fold_accuracies) / k_folds
    avg_val_loss = sum(fold_val_losses) / k_folds

    print(f"\nAverage Validation Accuracy across {k_folds} folds: {avg_accuracy:.2f}%")
    print(f"Average Validation Loss across {k_folds} folds: {avg_val_loss:.4f}")

    # Save final plots after all folds are complete
    plot_metrics(
        all_train_losses, all_val_losses, all_val_accuracies,
        output_file=f"{plots_dir}/training_metrics.png"
    )

    # Evaluate on the test set for confusion matrix and ROC curve
    print("Evaluating on the test set...")
    test_metrics = evaluate_model(model, test_loader, device)  # Use test_loader if available
    test_predictions = test_metrics["predictions"]
    test_labels = test_metrics["labels"]

    plot_confusion_matrix(
        test_labels, test_predictions, class_names=["Normal", "Pneumonia"],
        output_file=f"{plots_dir}/confusion_matrix.png"
    )
    plot_roc_curve(
        test_labels, test_predictions,
        output_file=f"{plots_dir}/roc_curve.png"
    )
    print("Final plots (ROC curve and confusion matrix) saved.")

    # Save fold results to a CSV file
    save_results(
        fold_accuracies, fold_val_losses, k_folds,
        output_file=f"{results_dir}/kfold_results.csv"
    )

    print("Training complete. Results and plots saved.")

    return fold_accuracies, fold_val_losses, all_train_losses, all_val_losses, all_val_accuracies

