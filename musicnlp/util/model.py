from typing import Union

from transformers import TransfoXLConfig, ReformerConfig


class MusicTransformerMixin:
    pass


def config2model_size(conf: Union[TransfoXLConfig, ReformerConfig]) -> int:
    assert isinstance(conf, (TransfoXLConfig, ReformerConfig))
    if isinstance(conf, TransfoXLConfig):
        return conf.d_model
    else:
        return conf.max_position_embeddings
