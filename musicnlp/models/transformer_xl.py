from typing import List, Tuple, Optional

import torch
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers.file_utils import ModelOutput

from musicnlp.models.models import MusicTransformerMixin


from typing import Dict, Any

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
        presets[k].update(dict(
            d_embed=hd_sz,  # saves a projection layer when hidden size is embedding size
            d_inner=hd_sz * 4,
            d_head=hd_sz // n_head,  # ensure dim_head x #head == hidden size
            mem_len=64 if 'debug' in k else 512,  # TODO: if i understand correctly this is segment length?
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
        return dict(n_layer=self.n_layer, hidden_size=self.d_embed, ff_size=self.d_inner)


class TransfoXLLMHeadModelOutput(ModelOutput):
    """
    Exactly the same as that in HF, but the one in HF is not available for import
    """

    losses: Optional[torch.FloatTensor] = None
    prediction_scores: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None

    @property
    def logits(self):
        return self.prediction_scores


class MyTransfoXLLMHeadModel(TransfoXLLMHeadModel, MusicTransformerMixin):
    """
    Modify so that logits is returned for better training monitoring

    Can do this since small vocab size
    """
    def forward(
        self,
        key_scores=None,  # not used, passed down for IKR computation
        input_ids=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
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
        if labels is None:
            prediction_scores = softmax_output.view(bsz, tgt_len, -1)
        else:
            # Run softmax again to get vocabulary logits
            # Not most efficient but less error-prone
            with torch.no_grad():
                sm_out = self.crit(pred_hid)
            prediction_scores = sm_out.view(bsz, tgt_len, -1)
        # ========================== End of modified ==========================

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