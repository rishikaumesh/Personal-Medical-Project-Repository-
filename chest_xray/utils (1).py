import matplotlib.pyplot as plt
import csv

def plot_metrics(train_losses, val_losses, val_accuracies, fold=None):
    """
    Plot training and validation metrics.

    Args:
        train_losses (list): Training losses over epochs.
        val_losses (list): Validation losses over epochs.
        val_accuracies (list): Validation accuracies over epochs.
        fold (int, optional): Current fold number for labeling. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss{' - Fold ' + str(fold) if fold else ''}")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy{' - Fold ' + str(fold) if fold else ''}")
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_results(fold_accuracies, fold_val_losses, k_folds, output_file="kfold_results.csv"):
    """
    Save fold results to a CSV file.

    Args:
        fold_accuracies (list): List of accuracies for each fold.
        fold_val_losses (list): List of validation losses for each fold.
        k_folds (int): Number of folds in the cross-validation.
        output_file (str): File name to save the results.

    Returns:
        None
    """
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Fold", "Validation Accuracy (%)", "Validation Loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for fold in range(k_folds):
            writer.writerow({
                "Fold": fold + 1,
                "Validation Accuracy (%)": fold_accuracies[fold],
                "Validation Loss": fold_val_losses[fold]
            })

    print(f"Results saved to {output_file}")
