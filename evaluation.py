import pandas as pd
import torch
import numpy as np


import torch
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def evaluate_model(model, test_loader, num_classes, results_csv="test_results.csv", device="cpu"):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Variables to track class-wise accuracy
    class_correct = [0] * num_classes  # Correct predictions per class
    class_total = [0] * num_classes    # Total samples per class

    # Confusion matrix initialization
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Track top-2 score if num_classes is 4
    top2_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Perform a forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update top-2 score if num_classes == 4
            if num_classes == 4:
                top2_preds = torch.topk(outputs, 2, dim=1).indices
                top2_correct += sum([labels[i].item() in top2_preds[i].tolist() for i in range(len(labels))])

            # Confusion matrix update
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                class_total[true_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1
                confusion_matrix[true_label][pred_label] += 1

    # Global accuracy calculation
    accuracy = 100 * correct / total

    # Class-wise accuracy calculation
    class_accuracies = [
        (100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0)
        for i in range(num_classes)
    ]

    # Calculate precision, recall, and F1-score per class
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in range(num_classes):
        TP = confusion_matrix[i][i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision * 100)
        recall_per_class.append(recall * 100)
        f1_per_class.append(f1 * 100)

    # Calculate global metrics
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    # Calculate Top-2 accuracy if applicable
    top2_accuracy = (100 * top2_correct / total) if num_classes == 4 else None

    # Save the results to a CSV file
    metrics = ["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1 Score"]
    values = [accuracy, macro_precision, macro_recall, macro_f1]

    if num_classes == 4:
        metrics.append("Top-2 Accuracy")
        values.append(top2_accuracy)

    # Add class-wise metrics
    metrics += [f"Accuracy Class {i}" for i in range(num_classes)] + \
               [f"Precision Class {i}" for i in range(num_classes)] + \
               [f"Recall Class {i}" for i in range(num_classes)] + \
               [f"F1 Score Class {i}" for i in range(num_classes)]

    values += class_accuracies + precision_per_class + recall_per_class + f1_per_class

    # Create DataFrame for metrics
    results = pd.DataFrame({"Metric": metrics, "Value": values})

    # Create DataFrame for confusion matrix
    confusion_df = pd.DataFrame(confusion_matrix, 
                                columns=[f"Pred {i}" for i in range(num_classes)], 
                                index=[f"True {i}" for i in range(num_classes)])

    # Save everything in a single CSV file
    with open(results_csv, "w") as f:
        results.to_csv(f, index=False)
        f.write("\n")  # Aggiunge una riga vuota
        confusion_df.to_csv(f)

    print(f"Test results saved to: {results_csv}")

    return accuracy, class_accuracies, precision_per_class, recall_per_class, f1_per_class, top2_accuracy




def validate_model(model, val_loader, device="cpu"):
    """
    Valuta il modello sul validation set.

    Args:
        model: Modello PyTorch.
        val_loader: DataLoader per il validation set.
        device: Dispositivo per l'addestramento (default: "cpu").

    Returns:
        val_accuracy: Accuratezza sul validation set.
    """
    model.eval()  # Modalità di valutazione
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    return val_accuracy


