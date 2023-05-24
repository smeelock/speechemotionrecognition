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


class FusionModelDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        fusion_strategy,
        tokenizer: PreTrainedTokenizerBase = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = 'pt'
    ):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)
        self.fusion_strategy = fusion_strategy

    def __call__(self, features):
        labels = torch.tensor([feature['label'] for feature in features], dtype=torch.long)
        rep1s = [torch.tensor(feature['rep1']).squeeze() for feature in features]
        rep2s = [torch.tensor(feature['rep2']).squeeze() for feature in features]

        # Pad sequences independently
        padded_rep1s = pad_sequence(rep1s, batch_first=True)  # [n_samples, time, embed_dim]
        padded_rep2s = pad_sequence(rep2s, batch_first=True)  # [n_samples, time, embed_dim]

        # Apply the fusion strategy
        if self.fusion_strategy == 'max_pooling':
            _pool = lambda x, size: F.adaptive_max_pool1d(x.permute(0, 2, 1), output_size=size)
            time_dim1 = padded_rep1s.size(1)
            time_dim2 = padded_rep2s.size(1)

            if time_dim1 == time_dim2:
                collated = torch.cat([padded_rep1s, padded_rep2s], dim=-1)
            elif time_dim1 > time_dim2:
                pooled_rep1 = _pool(padded_rep1s, time_dim2)
                pooled_rep1 = pooled_rep1.permute(0, 2, 1)
                collated = torch.cat([pooled_rep1, padded_rep2s], dim=-1)
            elif time_dim2 > time_dim1:
                pooled_rep2 = _pool(padded_rep2s, time_dim1)
                pooled_rep2 = pooled_rep2.permute(0, 2, 1)
                collated = torch.cat([padded_rep1s, pooled_rep2], dim=-1)

        else:
            raise ValueError("Invalid fusion strategy")

        batch = {
            'labels': labels,
            'input_values': collated,
        }
        return batch
