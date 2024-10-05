import numpy as np


def calculate_metrics(output_test, predictions):
    tp = np.sum((predictions == 1) & (output_test == 1))  # True Positives
    fp = np.sum((predictions == 1) & (output_test == 0))  # False Positives
    fn = np.sum((predictions == 0) & (output_test == 1))  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1