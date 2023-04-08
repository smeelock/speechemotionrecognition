import torch
from torchaudio.datasets import IEMOCAP

from .constants import DEFAULT_TARGET_SAMPLING_RATE


class ProcessedIEMOCAP(torch.utils.data.Dataset):
    def __init__(self, data: IEMOCAP, processor, config):
        self.data = data
        self.processor = processor
        self.config = config

        if hasattr(config, "target_sampling_rate"):
            self.target_sampling_rate = config.target_sampling_rate
        elif (
            hasattr(self.processor, "feature_extractor")
            and hasattr(self.processor.feature_extractor, "sampling_rate")
        ):
            self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        else:
            self.target_sampling_rate = DEFAULT_TARGET_SAMPLING_RATE

    def __getitem__(self, index):
        wav, _, _, label, _ = self.data[index]
        inputs = self.processor(wav.squeeze(), sampling_rate=self.target_sampling_rate)
        inputs["labels"] = self.config.label2id[label]
        return inputs

    def __len__(self):
        return len(self.data)
