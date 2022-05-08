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
            hidden_size=128, feed_forward_size=128 * 4, axial_pos_embds_dim=(32, 96),
            # note attention head size in config is per head
            num_attention_heads=8, attention_head_size=int(128 / 8),
            # effectively 6 layers as default config; going even smaller produces an error
            attn_layers=_layer_pair * 3
        ),
        'debug-large': dict(
            max_position_embeddings=512, axial_pos_shape=(16, 32),
            hidden_size=128, feed_forward_size=128 * 4, axial_pos_embds_dim=(32, 96),
            # note attention head size in config is per head
            num_attention_heads=8, attention_head_size=int(128 / 8),
            # effectively 6 layers as default config; going even smaller produces an error
            attn_layers=_layer_pair * 3
        ),
        # overall, given hidden size, keep
        #   feed_forward_size = 4 x hidden size
        #   attention_head_size = hidden_size
        'tiny': dict(
            max_position_embeddings=1024, axial_pos_shape=(32, 32),
            hidden_size=256, feed_forward_size=256 * 4, axial_pos_embds_dim=(64, 192),
            # note attention head size in config is per head
            num_attention_heads=8, attention_head_size=int(256 / 8),
            # effectively 6 layers as default config; going even smaller produces an error
            attn_layers=_layer_pair * 3
        ),
        'small': dict(
            max_position_embeddings=2048, axial_pos_shape=(32, 64),
            hidden_size=512, feed_forward_size=512 * 4, axial_pos_embds_dim=(128, 512 - 128),
            num_attention_heads=8, attention_head_size=int(512 / 8),
            attn_layers=_layer_pair * 3
        ),
        'base': dict(
            max_position_embeddings=2048, axial_pos_shape=(32, 64),
            hidden_size=768, feed_forward_size=768 * 4, axial_pos_embds_dim=(192, 768 - 192),
            num_attention_heads=12, attention_head_size=int(768 / 12),
            attn_layers=_layer_pair * 6,
            num_hashes=2  # for better accuracy
        ),
        'large': dict(
            max_position_embeddings=2048, axial_pos_shape=(32, 64),  # TODO: support token length 4096?
            hidden_size=1024, feed_forward_size=1024 * 4, axial_pos_embds_dim=(256, 1024 - 256),
            num_attention_heads=16, attention_head_size=int(1024 / 16),
            attn_layers=_layer_pair * 12,
            num_hashes=2
        )
    }

    def __init__(self, model_size: str = 'base', tokenizer: MusicTokenizer = None, **kwargs):
        config = MyReformerConfig.presets[model_size]
        if tokenizer:
            # In normal model loading, a tokenizer should always be passed in
            # the omission is for HF saving model only, where the fields are already saved,
            #   see `PretrainedConfig.to_diff_dict`
            config.update(dict(  # default config for any reformer config
                is_decoder=True,
                num_buckets=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size
            ))
        config.update(kwargs)
        super().__init__(**config)

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
