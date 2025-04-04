import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred, labels):
    preds = np.argmax(pred, axis=-1)

    accuracy = accuracy_score(labels, preds)
    weighted_f1 = f1_score(labels, preds, average='weighted')

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1
    }