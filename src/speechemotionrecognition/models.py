from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import WhisperPreTrainedModel, WhisperModel
from transformers.utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SpeechClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.merge if hasattr(config, "merge") else "max"

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def merge_inputs(self, hidden_states):
        if self.mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif self.mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif self.mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
        return outputs

    def forward(self, input_values, labels=None, return_dict=None):
        x = self.merge_input(input_values)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + input_values[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=input_values.hidden_states,
            attentions=input_values.attentions,
        )


class WhisperEncoderAsFeatureExtractor(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = WhisperModel(config).encoder

        # only keep first n encoding layers
        self.feature_extractor.layers = self.feature_extractor.layers[:config.num_encoder_layers]

        self.init_weights()

    def forward(self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
                ):
        return self.feature_extractor(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
