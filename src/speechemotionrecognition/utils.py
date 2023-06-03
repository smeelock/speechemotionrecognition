import numpy as np
from transformers import EvalPrediction


def get_compute_metrics(metrics):
    def _compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {k: metric(p.label_ids, preds) for k, metric in metrics.items()}

    return _compute_metrics


def clean_model_name(model_name):
    return model_name.split('/')[-1]


def get_fusion_model_embed_dim(dataset, data_collator):
    examples = [dataset[0], dataset[1], dataset[2]]
    collated = data_collator(examples)
    return collated['input_values'].size(-1)


def get_cv_splits(dataset, n_cv_groups=3):
    speakers = np.unique(dataset["speaker"])
    indices = np.arange(len(dataset))
    for i in range(0, len(speakers), n_cv_groups):
        test_speakers = speakers[i:i + n_cv_groups]
        test_mask = np.isin(dataset["speaker"], test_speakers, assume_unique=True)
        train_mask = ~test_mask
        yield indices[train_mask], indices[test_mask]
