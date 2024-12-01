import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to compute average accuracy and loss from a CSV file
def summarize_kfold_results(csv_file):
    # Load the CSV file
    results = pd.read_csv(csv_file)

    # Dynamically handle column names
    accuracy_col = None
    loss_col = None

    # Check for column names dynamically
    for col in results.columns:
        if "Accuracy" in col or "accuracy" in col:
            accuracy_col = col
        if "Loss" in col or "loss" in col:
            loss_col = col

    if not accuracy_col or not loss_col:
        raise KeyError(f"Columns for accuracy and loss not found in {csv_file}.")

    avg_accuracy = results[accuracy_col].mean()
    avg_loss = results[loss_col].mean()
    return avg_accuracy, avg_loss

# List of models and their corresponding CSV file names
models = {
    "PneumoniaDetectionCNN": "kfold_results.csv",
    "PneumoniaDetectionCNN_v2": "kfold_results2.csv",
    "ImprovedPneumoniaDetectionCNN": "kfold_results3.csv",
    "ImprovedPneumoniaDetectionCNN3": "kfold_results4.csv"
}

# Summarize results for each model
summary = []
for model_name, csv_file in models.items():
    try:
        avg_accuracy, avg_loss = summarize_kfold_results(csv_file)
        summary.append({
            "Model": model_name,
            "Average Accuracy (%)": avg_accuracy,
            "Average Validation Loss": avg_loss
        })
    except FileNotFoundError:
        print(f"File not found: {csv_file}. Skipping this model.")
    except KeyError as e:
        print(f"KeyError: {e}. Check the column names in the CSV file {csv_file}.")

# Convert summary to DataFrame for better display
summary_df = pd.DataFrame(summary)
print(summary_df)


# Plot Average Accuracy
plt.figure(figsize=(12, 6))
plt.bar(summary_df["Model"], summary_df["Average Accuracy (%)"], color='skyblue')
plt.title("Average Accuracy Across K-Fold for Each Model")
plt.ylabel("Average Accuracy (%)")
plt.xticks(rotation=45)
plt.savefig("plots/average_accuracy.png")
print("Average accuracy plot saved as plots/average_accuracy.png")
plt.close()

# Plot Average Validation Loss
plt.figure(figsize=(12, 6))
plt.bar(summary_df["Model"], summary_df["Average Validation Loss"], color='salmon')
plt.title("Average Validation Loss Across K-Fold for Each Model")
plt.ylabel("Average Validation Loss")
plt.xticks(rotation=45)
plt.savefig("plots/average_validation_loss.png")
print("Average validation loss plot saved as plots/average_validation_loss.png")
plt.close()
