from typing import List, Tuple, Optional

import torch
from transformers import TransfoXLLMHeadModel
from transformers.file_utils import ModelOutput

from musicnlp.models.models import MusicTransformerMixin


# TODO
"""

    xl={
        'debug': dict(d_model=8),
        'debug-large': dict(d_model=512),
        'small': dict(d_model=1024)
    },
    """


class MyTransfoXLLMHeadModelOutput(ModelOutput):
    # ========================== Begin of Modified ==========================
    loss: Optional[torch.FloatTensor] = None
    # losses: Optional[torch.FloatTensor] = None
    # ========================== End of Modified ==========================
    prediction_scores: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def logits(self):
        return self.prediction_scores


class MyTransfoXLLMHeadModel(TransfoXLLMHeadModel, MusicTransformerMixin):
    """
    For compatibility with huggingface Trainer API
    See https://github.com/huggingface/transformers/issues/11822#issuecomment-847614016
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self, input_ids=None, mems=None, head_mask=None, inputs_embeds=None, labels=None,
            output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        elif inputs_embeds is not None:
            bsz, tgt_len = inputs_embeds.size(0), inputs_embeds.size(1)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        transformer_outputs = self.transformer(
            input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]

        softmax_output = self.crit(pred_hid, labels)
        prediction_scores = softmax_output.view(bsz, tgt_len, -1) if labels is None else ()
        # ========================== Begin of added ==========================
        # hf implementation doesn't return global logits when `labels` given, i.e. in training mode
        #   cos compute in segments?
        # TODO: pretty ugly to get the logits, as `ProjectedAdaptiveLogSoftmax` is not exported
        # ========================== End of added ==========================
        loss = softmax_output.view(bsz, tgt_len - 1) if labels is not None else None
        # ========================== Begin of added ==========================
        loss = loss.mean()
        # ========================== End of added ==========================

        if not return_dict:
            output = (prediction_scores,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MyTransfoXLLMHeadModelOutput(
            # ========================== Begin of modified ==========================
            loss=loss,
            # losses=loss,
            # ========================== End of modified ==========================
            prediction_scores=prediction_scores,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
