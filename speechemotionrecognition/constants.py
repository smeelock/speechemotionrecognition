import os

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# paths
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PACKAGE_DIR, "data/raw")
DEFAULT_OUTPUT_DIR = os.path.join(PACKAGE_DIR, "runs/")
DEFAULT_CACHE_DIR = os.path.join(PACKAGE_DIR, "cache/")
DEFAULT_LOGS_DIR = os.path.join(PACKAGE_DIR, "logs/")

# wandb
DEFAULT_WANDB_WATCH = "all"
DEFAULT_WANDB_LOG_MODEL = "checkpoint"
DEFAULT_WANDB_PROJECT = "huggingface"

# dataset
DEFAULT_TARGET_SAMPLING_RATE = 16000
DEFAULT_IEMOCAP_DIR = os.path.join(DEFAULT_DATA_DIR, "IEMOCAP")
DEFAULT_IEMOCAP_LABEL_LIST = ["neu", "hap", "ang", "sad", "exc"]
DEFAULT_IEMOCAP_LABEL2ID = {label: i for i, label in enumerate(DEFAULT_IEMOCAP_LABEL_LIST)}
DEFAULT_IEMOCAP_ID2LABEL = {i: label for i, label in enumerate(DEFAULT_IEMOCAP_LABEL_LIST)}

# training
DEFAULT_DEBUG_SIZE = 0.1  # 0.1 = 10% of the dataset
DEFAULT_TEST_SPLIT_SIZE = 0.2
DEFAULT_SEED = 42

# evaluation
DEFAULT_METRICS = {
    "unweighted_accuracy": accuracy_score,
    "weighted_accuracy": balanced_accuracy_score,
    "micro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
    "macro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
}

# models
DEFAULT_WHISPER_MODEL_NAME = "openai/whisper-base"
DEFAULT_WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
