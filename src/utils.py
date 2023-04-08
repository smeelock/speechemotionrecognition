import numpy as np
from transformers import EvalPrediction

from constants import DEFAULT_METRICS


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {k: metric(p.label_ids, preds) for k, metric in DEFAULT_METRICS.items()}
