from typing import Dict, Any

import numpy as np
from transformers import ReformerConfig

from musicnlp.vocab import MusicTokenizer


__all__ = ['MyReformerConfig']


class MyReformerConfig(ReformerConfig):
    _layer_pair = ['local', 'lsh']
    presets = {
        'debug': dict(
            max_position_embeddings=64, axial_pos_shape=(8, 8),
            hidden_size=128, num_attention_heads=8,  # attention head size in config is per head
            # effectively 6 layers as default config; going even smaller produces an error
            attn_layers=_layer_pair*3
        ),
        'debug-large': dict(
            max_position_embeddings=512, axial_pos_shape=(16, 32),
            hidden_size=128, num_attention_heads=8, attn_layers=_layer_pair*3
        ),
        'tiny': dict(
            max_position_embeddings=1024, axial_pos_shape=(32, 32),
            hidden_size=256, num_attention_heads=8, attn_layers=_layer_pair*3
        ),
        'small': dict(
            max_position_embeddings=2048, axial_pos_shape=(32, 64),
            hidden_size=512, num_attention_heads=8, attn_layers=_layer_pair*3
        ),
        'base': dict(
            max_position_embeddings=2048, axial_pos_shape=(32, 64),
            hidden_size=768, num_attention_heads=12, attn_layers=_layer_pair*6,
            num_hashes=2  # for better accuracy
        ),
        'large': dict(
            max_position_embeddings=2048, axial_pos_shape=(32, 64),  # TODO: support token length 4096?
            hidden_size=1024, num_attention_heads=16, attn_layers=_layer_pair*12,
            num_hashes=2
        )
    }
    for k, d_config in presets.items():
        hd_sz, n_head = d_config['hidden_size'], d_config['num_attention_heads']
        assert hd_sz % n_head == 0
        assert hd_sz % 4 == 0  # for splitting the axial positional embedding
        presets[k].update(dict(
            feed_forward_size=hd_sz * 4,
            attention_head_size=hd_sz // n_head,
            axial_pos_embds_dim=(hd_sz // 4, 3 * hd_sz // 4),
            is_decoder=True,
            num_buckets=None  # have reformer decide
        ))

    def __init__(self, model_size: str = 'base', tokenizer: MusicTokenizer = None, **kwargs):
        d_config = MyReformerConfig.presets[model_size]
        if tokenizer:
            # In normal model loading, a tokenizer should always be passed in
            # the omission is for HF saving model only, where the fields are already saved,
            #   see `PretrainedConfig.to_diff_dict`
            d_config.update(dict(  # default config for any reformer config
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size
            ))
        d_config.update(kwargs)
        super().__init__(**d_config)

        aps, mpe = self.axial_pos_shape, self.max_position_embeddings
        assert len(aps) == 2 and np.prod(aps) == mpe, \
            'the product of `axial_pos_shape` must be `max_position_embeddings`'

    @property
    def model_max_length(self) -> int:
        return self.max_position_embeddings

    @property
    def model_meta(self) -> Dict[str, Any]:
        return dict(
            axial_pos_shape=self.axial_pos_shape,
            n_layer=len(self.attn_layers),
            hidden_size=self.hidden_size, ff_size=self.feed_forward_size,
            attention_shape=f'{self.num_attention_heads}x{self.attention_head_size}',
        )
