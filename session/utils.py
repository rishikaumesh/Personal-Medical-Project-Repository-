import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import csv

def plot_metrics(train_losses, val_losses, val_accuracies, output_file="plots/training_metrics.png"):
    """
    Plot and save training/validation loss and accuracy curves.
    """
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Training metrics saved at: {output_file}")
    plt.close()


def plot_roc_curve(labels, predictions, output_file="plots/roc_curve.png"):
    """
    Plot and save the ROC curve.
    """
    os.makedirs("plots", exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    print(f"ROC curve saved at: {output_file}")
    plt.close()


def plot_confusion_matrix(labels, predictions, class_names, output_file="plots/confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    """
    os.makedirs("plots", exist_ok=True)

    conf_matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)
    print(f"Confusion matrix saved at: {output_file}")
    plt.close()


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
