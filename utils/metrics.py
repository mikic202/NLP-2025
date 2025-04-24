import sklearn.metrics
import torch
import sklearn
from seaborn import heatmap


def display_clasification_metrics(predictions, targets):
    print(f"F1 Score {sklearn.metrics.f1_score(targets, predictions, average='macro')}")
    print(f"Accuracy {torch.sum(targets == predictions).item()/len(targets)}")
    print(
        f"Recall {sklearn.metrics.recall_score(targets, predictions, average='macro')}"
    )

    conf_matrix = sklearn.metrics.confusion_matrix(targets, predictions)
    heatmap(conf_matrix, annot=True, fmt="d")
