from dataclasses import dataclass
from typing import Union, Optional, List, Dict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase, Wav2Vec2Processor, \
    WhisperProcessor
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorCTCWithPadding:
    processor: Union[Wav2Vec2Processor, WhisperProcessor]
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        label_features = [feature["label"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        return batch


class FusionDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = 'pt'
    ):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)

    def prepare(self, features):
        rep1s = [torch.tensor(feature['rep1']).squeeze() for feature in features]
        rep2s = [torch.tensor(feature['rep2']).squeeze() for feature in features]

        # Pad sequences independently
        r1 = pad_sequence(rep1s, batch_first=True)  # [n_samples, time, embed_dim]
        r2 = pad_sequence(rep2s, batch_first=True)  # [n_samples, time, embed_dim]

        # Apply max_pooling fusion strategy
        _pool = lambda x, size: F.adaptive_max_pool1d(x.permute(0, 2, 1), output_size=size)
        time_dim1 = r1.size(1)
        time_dim2 = r2.size(1)

        if time_dim1 > time_dim2:
            r1 = _pool(r1, time_dim2).permute(0, 2, 1)
        elif time_dim2 > time_dim1:
            r2 = _pool(r2, time_dim1).permute(0, 2, 1)

        return r1, r2


class BaselineFusionModelDataCollator(FusionDataCollator):
    def __call__(self, features):
        labels = torch.tensor([feature['label'] for feature in features], dtype=torch.long)

        # prepare representations
        r1, r2 = self.prepare(features)

        # concat representations for baseline fusion model
        collated = torch.cat((r1, r2), dim=-1)

        batch = {
            'labels': labels,
            'input_values': collated,
        }
        return batch


class FusionModelDataCollator(FusionDataCollator):
    def __call__(self, features):
        labels = torch.tensor([feature['label'] for feature in features], dtype=torch.long)

        # prepare representations
        r1, r2 = self.prepare(features)

        # concat representations for baseline fusion model
        collated = torch.cat((r1, r2), dim=-1)

        # fix embedding dimensions of rep1 & rep2
        embed_dim1 = r1.size(2)
        embed_dim2 = r2.size(2)
        if embed_dim1 > embed_dim2:
            r2 = torch.tile(r2, (1, 1, embed_dim1 // embed_dim2))
        elif embed_dim2 > embed_dim1:
            r1 = torch.tile(r1, (1, 1, embed_dim2 // embed_dim1))
        assert r1.size(2) == r2.size(2)

        batch = {
            'labels': labels,
            'input_values': (r1, r2)  # tuple
        }
        return batch
