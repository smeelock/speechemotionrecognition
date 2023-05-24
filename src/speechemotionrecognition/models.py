from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import WhisperPreTrainedModel, WhisperModel, PreTrainedModel, PretrainedConfig
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
        self.config = config
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

    def forward(self, input_values, labels=None):
        x = self.merge_inputs(input_values)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits
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


@dataclass
class FusionModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class FusionConfig(PretrainedConfig):
    def __init__(self, embed_dim=1152, hidden_dim=512, num_classes=4, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes


class BaselineFusionModel(PreTrainedModel):
    def __init__(self, config):
        super(BaselineFusionModel, self).__init__(config)
        self.self_attention = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=config.embed_dim, out_features=config.hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=config.hidden_dim, out_features=config.num_classes),
        )
        self.init_weights()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,  # labels are needed for loss computation
    ) -> Union[Tuple, FusionModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if hasattr(input_values, "last_hidden_state"):
            input_values = input_values['last_hidden_state']

        attended, attn_weights = self.self_attention(input_values, input_values, input_values)

        hidden_states = attended.clone()  # Cloning to ensure it doesn't affect subsequent computations
        attended = attended.mean(dim=1)  # mean pooling on time dimension
        logits = self.classifier(attended)  # classify without softmax

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_classes), labels.view(-1))

        if not return_dict:
            return_tuple = (logits,)
            if output_hidden_states:
                return_tuple = return_tuple + (hidden_states,)
            if output_attentions:
                return_tuple = return_tuple + (attn_weights,)
            return return_tuple

        return FusionModelOutput(
            loss=loss,
            last_hidden_state=logits,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attn_weights if output_attentions else None
        )
