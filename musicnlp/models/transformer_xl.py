from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers.modeling_utils import ModelOutput

from musicnlp.vocab import MusicTokenizer


__all__ = ['MyTransfoXLConfig', 'MyTransfoXLLMHeadModel']


class MyTransfoXLConfig(TransfoXLConfig):
    presets = {
        'debug': dict(d_model=128, n_head=8, n_layer=4),
        'debug-large': dict(d_model=128, n_head=8, n_layer=4),
        'tiny': dict(d_model=256, n_head=8, n_layer=6),
        'small': dict(d_model=512, n_head=8, n_layer=6),
        'base': dict(d_model=768, n_head=12, n_layer=12),
        'large': dict(d_model=1024, n_head=16, n_layer=18)
    }
    size2max_length = {'debug': 64, 'debug-large': 128, 'tiny': 512, 'small': 1024, 'base': 2048, 'large': 2048}

    for k, d_config in presets.items():
        hd_sz, n_head = d_config['d_model'], d_config['n_head']
        assert hd_sz % n_head == 0

        if 'debug' in k:
            m_len, c_len = 64, 64
        else:
            m_len = max(128, size2max_length[k] // 8)
            c_len = max(1024, size2max_length[k] // 2)
        presets[k].update(dict(
            d_embed=hd_sz,  # saves a projection layer when hidden size is embedding size
            d_inner=hd_sz * 4,
            d_head=hd_sz // n_head,  # ensure dim_head x #head == hidden size
            mem_len=m_len,  # TODO: if i understand correctly this is segment length?
            clamp_len=c_len,
            # intended that adaptive softmax is effectively not needed, given the small Music vocab size
            div_val=1, cutoffs=[]
        ))
        # Don't understand `proj_share_all_but_first` and it's not used in modeling
        # `adaptive` is not really configurable

    def __init__(self, model_size: str = 'base', tokenizer: MusicTokenizer = None, max_length: int = None, **kwargs):
        config = MyTransfoXLConfig.presets[model_size]
        if tokenizer:  # same argument as in `reformer`
            config['vocab_size'] = tokenizer.vocab_size
        config.update(kwargs)
        super().__init__(**config)
        # still fix a cut-off for training memory cap
        self.max_length_ = max_length or MyTransfoXLConfig.size2max_length[model_size]

    @property
    def model_meta(self) -> Dict[str, Any]:
        return dict(
            n_layer=self.n_layer, hidden_size=self.d_embed, ff_size=self.d_inner,
            seg_len=self.mem_len, max_len=self.max_length_
        )


# Taken completely from HF to override TransformerXL return
@dataclass
class TransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (`torch.FloatTensor` of shape *(batch_size, sequence_length-1)*, *optional*, returned when `labels` is provided):
            Language modeling losses (not reduced).
        prediction_scores (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is provided)
            Reduced language modeling loss.
    """

    losses: Optional[torch.FloatTensor] = None
    prediction_scores: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None

    @property
    def logits(self):
        # prediction scores are the output of the adaptive softmax, see
        # the file `modeling_transfo_xl_utilities`. Since the adaptive
        # softmax returns the log softmax value, `self.prediction_scores`
        # are strictly speaking not exactly `logits`, but behave the same
        # way logits do.
        return self.prediction_scores


class MyTransfoXLLMHeadModel(TransfoXLLMHeadModel):
    cls_name = 'TransformerXl'

    def forward(
            self,
            # ========================== Begin of added ==========================
            # pass in `key_scores` for eval metrics
            key_scores=None,
            # ========================== End of added ==========================
            input_ids: Optional[torch.LongTensor] = None,
            mems: Optional[List[torch.FloatTensor]] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
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

        if labels is not None:
            # Prevents all labels being -100 and throwing an error
            # when backwarding the loss
            miss_valid_label = labels[0, 1:].sum() == (labels.size(1) - 1) * -100
            if miss_valid_label:
                # Sets an <EOS> token, just to prevent loss from being NaN
                labels[0, 1] = self.config.eos_token_id

        softmax_output = self.crit(pred_hid, labels)
        # ========================== Begin of modified ==========================
        # prediction_scores = softmax_output.view(bsz, tgt_len, -1) if labels is None else ()
        # To get logits and hence NTP ACC during eval
        _softmax_output = softmax_output
        in_eval = not self.training
        if in_eval:
            if labels is not None:  # re-run attention to get vocab-length logits and hence token prediction for metrics
                _softmax_output = self.crit(pred_hid, None)
        prediction_scores = _softmax_output.view(bsz, tgt_len, -1) if (labels is None or in_eval) else ()
        # ========================== Begin of modified ==========================

        if labels is not None:
            losses = softmax_output.view(bsz, tgt_len - 1)
            # Avoids from incorporating padding (-100) tokens into loss value
            loss = losses[losses != 0].mean()
        else:
            losses, loss = None, None

        if not return_dict:
            if self.trainer_compatible:
                output = (prediction_scores, losses) if losses is not None else (prediction_scores,)
                output += transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
                output = (prediction_scores, *transformer_outputs[1:])
                output = ((losses,) + output) if losses is not None else output
                return (output + (loss,)) if loss is not None else output

        return TransfoXLLMHeadModelOutput(
            loss=loss,
            prediction_scores=prediction_scores,
            losses=losses,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
