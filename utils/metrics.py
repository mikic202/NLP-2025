import sklearn.metrics
import torch
import sklearn
from seaborn import heatmap


def get_metrics(predictions, targets):
    return {
        "f1_score": sklearn.metrics.f1_score(targets, predictions, average="macro"),
        "accuracy": torch.sum(targets == predictions).item() / len(targets),
        "recall": sklearn.metrics.recall_score(targets, predictions, average="macro"),
        "confusion_matrix": sklearn.metrics.confusion_matrix(targets, predictions),
    }


def display_clasification_metrics(predictions, targets):
    f1_score, accuracy, recall, conf_matrix = get_metrics(predictions, targets).values()

    print(f"F1 Score {f1_score}")
    print(f"Accuracy {accuracy}")
    print(f"Recall {recall}")
    heatmap(conf_matrix, annot=True, fmt="d")
