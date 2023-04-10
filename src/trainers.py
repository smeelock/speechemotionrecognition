from dataclasses import dataclass
from typing import Union, Optional, List, Dict

import torch
from torch.cuda.amp import autocast
from transformers import Trainer, ProcessorMixin, is_apex_available

if is_apex_available():
    from apex import amp


@dataclass
class DataCollatorCTCWithPadding:
    processor: ProcessorMixin
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


class CTCTrainer(Trainer):
    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if hasattr(self, 'use_amp') and self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if hasattr(self, 'use_amp') and self.use_amp:
            self.scaler.scale(loss).backward()
        elif hasattr(self, 'use_apex') and self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif hasattr(self, 'deepspeed') and self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.sum().backward()

        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        model.eval()
        inputs = self._prepare_inputs(inputs)

        labels = inputs.get("labels")

        with autocast():
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = outputs.get("loss")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if hasattr(self, 'use_amp') and self.use_amp:
            self.scaler.scale(loss).backward()
        elif hasattr(self, 'use_apex') and self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif hasattr(self, 'deepspeed') and self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.sum().backward()

        with torch.no_grad():
            torch.cuda.empty_cache()

        if prediction_loss_only:
            return loss.detach()
        return (loss.detach(), logits.detach(), labels.detach())
