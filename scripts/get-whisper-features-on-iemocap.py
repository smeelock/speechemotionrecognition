"""
A script to get the features from the IEMOCAP dataset using OpenAI's whisper model as a feature extractor.

Original file is located at
    https://colab.research.google.com/drive/1-vw1bNt-e8DdzAk74tEqbIkBiL2hfaOf
"""
import os

import torch
import wandb
from torch.utils.data import Subset
from torchaudio.datasets import IEMOCAP  # https://pytorch.org/audio/master/generated/torchaudio.datasets.IEMOCAP.html
from transformers import WhisperPreTrainedModel, AutoConfig, WhisperProcessor

# ========= Configuration =========
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data/raw")

debug_size = 0.1  # 0.1 = 10% of the dataset

feature_to_idx = {key: i for i, key in enumerate(["wav", "sampling_rate", "filename", "label", "speaker"])}
label_list = ["neu", "hap", "ang", "sad", "exc", "fru"]
num_labels = len(label_list)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# ========= Model =========
model_name_or_path = "openai/whisper-base"
num_encoder_layers = 4
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
setattr(config, "num_encoder_layers", num_encoder_layers)


class FeatureExtractor(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # only keep some encoder layers
        self.whisper = WhisperPreTrainedModel(config).encoder
        self.whiser._freeze_parameters()
        self.whisper.layers = self.whisper.layers[:config.num_encoder_layers]

        self.init_weights()


model = FeatureExtractor.from_pretrained(
    model_name_or_path,
    config=config
)

# ========= IEMOCAP =========
processor = WhisperProcessor.from_pretrained(model_name_or_path)
target_sampling_rate = processor.feature_extractor.sampling_rate


class CustomIEMOCAP(torch.utils.data.Dataset):
    def __init__(self, data, processor, encoder):
        self.data = data
        self.processor = processor
        self.encoder = encoder

    def __getitem__(self, index):
        wav, _, _, label, _ = self.data[index]
        processed_inputs = self.processor(wav.squeeze(), sampling_rate=target_sampling_rate)
        encoded_inputs = self.encoder(**processed_inputs)
        encoded_inputs["labels"] = label2id[label]
        return encoded_inputs

    def __len__(self):
        return len(self.data)


iemocap = IEMOCAP(root=data_path)  # in function, path = root / "IEMOCAP"
iemocap = Subset(iemocap, range(int(debug_size * len(iemocap))))


# ========= Extract features =========
dataset = CustomIEMOCAP(data=iemocap, processor=processor, encoder=model)

# ========= Save features =========
torch.save(dataset, os.path.join(base_dir, "data/processed/whisper-encoded-iemocap-features.pt"))

with wandb.init(project="huggingface") as run:
    artifact = wandb.Artifact("whisper-encoded-iemocap-features", type="dataset")
    artifact.add_file(os.path.join(base_dir, "data/processed/whisper-encoded-iemocap-features.pt"))
    run.log_artifact(artifact)

