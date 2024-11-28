
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, dataloader, device, output_dir="plots", class_names=["Normal", "Pneumonia"]):
    """
    Evaluate the model and save confusion matrix and ROC curve.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test data.
        device (torch.device): Device to run the evaluation on.
        output_dir (str): Directory to save the evaluation plots.
        class_names (list): Names of the classes for the confusion matrix.

    Returns:
        dict: Dictionary containing evaluation metrics (precision, recall, F1 score, accuracy).
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")

    # Save confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved at: {confusion_matrix_path}")
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    print(f"ROC curve saved at: {roc_curve_path}")
    plt.close()

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


