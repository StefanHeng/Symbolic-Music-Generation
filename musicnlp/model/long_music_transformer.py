"""
Proposed method: Transformer-XL on compact melody & bass representation for music generation
"""
from typing import Dict, Optional, Tuple, List, Union

from tokenizers import AddedToken
from transformers import TensorType, BatchEncoding
from transformers.file_utils import PaddingStrategy

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TextInput, PreTokenizedInput, EncodedInput, \
    TruncationStrategy, TextInputPair, PreTokenizedInputPair, EncodedInputPair

from musicnlp.preprocess import MusicVocabulary


class LMTTokenizer(PreTrainedTokenizerBase):
    """
    Conversion between music tokens & int ids

    For integration with HuggingFace
    """
    def __init__(self, prec: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.vocab = MusicVocabulary(prec=prec, color=False)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        pass

    def get_vocab(self) -> Dict[str, int]:
        pass

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        pass

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        pass

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        pass

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput, EncodedInput],
                     text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
                     add_special_tokens: bool = True, padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
                     truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
                     max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False,
                     pad_to_multiple_of: Optional[int] = None, return_tensors: Optional[Union[str, TensorType]] = None,
                     return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None,
                     return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False,
                     return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True,
                     **kwargs) -> BatchEncoding:
        pass

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[
        List[TextInput],
        List[TextInputPair],
        List[PreTokenizedInput],
        List[PreTokenizedInputPair],
        List[EncodedInput],
        List[EncodedInputPair],
    ], add_special_tokens: bool = True, padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
                           truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
                           max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False,
                           pad_to_multiple_of: Optional[int] = None,
                           return_tensors: Optional[Union[str, TensorType]] = None,
                           return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None,
                           return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False,
                           return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True,
                           **kwargs) -> BatchEncoding:
        pass

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        pass

    def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False,
                clean_up_tokenization_spaces: bool = True, **kwargs) -> str:
        pass
