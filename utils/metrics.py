import sklearn.metrics
import torch
import sklearn
from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt


def get_metrics(predictions, targets, rule_based=False):
    if rule_based:
        accuracy = sklearn.metrics.accuracy_score(targets, predictions)
    else: 
        accuracy = torch.sum(targets == predictions).item() / len(targets)
    return {
        "f1_score": sklearn.metrics.f1_score(targets, predictions, average="macro"),
        "accuracy": accuracy,
        "recall": sklearn.metrics.recall_score(targets, predictions, average="macro"),
        "confusion_matrix": sklearn.metrics.confusion_matrix(targets, predictions),
    }


def display_clasification_metrics(predictions, targets, labels=None, rule_based=False):
    f1_score, accuracy, recall, conf_matrix = get_metrics(predictions, targets, rule_based).values()
    if labels is not None:
        labels = np.unique(targets)

    print(f"F1 Score {f1_score}")
    print(f"Accuracy {accuracy}")
    print(f"Recall {recall}")
    if rule_based:
        cm = sklearn.metrics.confusion_matrix(targets, predictions)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='rocket')
        # plt.show()
    else:
        heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
        )
