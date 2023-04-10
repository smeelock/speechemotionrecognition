import numpy as np
from transformers import EvalPrediction


def get_compute_metrics(metrics):
    def _compute_metrics(p: EvalPrediction):
        print('*'*30, "metrics!")
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {k: metric(p.label_ids, preds) for k, metric in metrics.items()}
    return _compute_metrics