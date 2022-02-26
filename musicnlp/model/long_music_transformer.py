"""
Proposed method: Transformer-XL on compact melody & bass representation for music generation
"""
import sys
from typing import Optional

import transformers
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback
from transformers.training_args import OptimizerNames
# from transformers import TensorType, BatchEncoding
# from transformers.file_utils import PaddingStrategy
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy
# from transformers.tokenization_utils_base import TextInputPair, PreTokenizedInputPair, EncodedInputPair
import datasets
from tokenizers import AddedToken
import torch

from musicnlp.util import *
from musicnlp.preprocess import MusicVocabulary


PT_LOSS_PAD = -100


class LMTTokenizer(PreTrainedTokenizer):
    """
    Conversion between music tokens & int ids

    For integration with HuggingFace

    Note that there are **no special tokens**
    """
    TOK_PAD = '[PAD]'

    model_input_names = ['input_ids']  # Per `TransfoXLTokenizer`

    def __init__(self, prec: int = 5, **kwargs):
        super().__init__(**kwargs)
        # Model max length undefined, for infinite input length; See `tokenization_utils_base`
        if self.model_max_length == int(1e30):
            self.model_max_length = 1024  # TODO: subject to change?

        self.prec = prec
        self.vocab = MusicVocabulary(prec=prec, color=False)
        self.spec_toks_enc, self.spec_toks_dec = dict(), dict()
        self._add_special_token(LMTTokenizer.TOK_PAD)
        self.pad_token = LMTTokenizer.TOK_PAD
        ic(self.pad_token, self.pad_token_id)

    def _add_special_token(self, tok):
        assert tok not in self.spec_toks_enc
        id_ = self.vocab_size
        self.spec_toks_enc[tok] = id_  # Assign the next coming value; vocab size automatically increments
        self.spec_toks_dec[id_] = tok

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise ValueError('In LMTTokenizer._add_tokens')

    def get_vocab(self) -> Dict[str, int]:
        raise ValueError('In LMTTokenizer.get_vocab')

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise ValueError('In LMTTokenizer.save_vocabulary')

    def _tokenize(self, text, **kwargs):
        # raise ValueError('In LMTTokenizer._tokenize')
        return text.split()  # Each word in vocab is

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.spec_toks_enc)

    def _convert_token_to_id(self, token):
        # raise ValueError('In LMTTokenizer._convert_token_to_id')
        return self.spec_toks_enc[token] if token in self.spec_toks_enc else self.vocab.t2i(token)

    def _convert_id_to_token(self, index: int) -> str:
        # raise ValueError('In LMTTokenizer._convert_id_to_token')
        return self.spec_toks_dec[index] if index in self.spec_toks_dec else self.vocab.i2t(index)
    #
    # def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
    #     raise ValueError('In LMTTokenizer.tokenize')
    #
    # def num_special_tokens_to_add(self, pair: bool = False) -> int:
    #     raise ValueError('In LMTTokenizer.num_special_tokens_to_add')
    #
    # def _encode_plus(self, text: Union[TextInput, PreTokenizedInput, EncodedInput],
    #                  text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    #                  add_special_tokens: bool = True, padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    #                  truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    #                  max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False,
    #                  pad_to_multiple_of: Optional[int] = None, return_tensors: Optional[Union[str, TensorType]] = None,
    #                  return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None,
    #                  return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False,
    #                  return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True,
    #                  **kwargs) -> BatchEncoding:
    #     # raise ValueError('In LMTTokenizer._encode_plus')
    #     text: List[str] = text.split()
    #     input_ids = self.vocab.encode(text)
    #     if padding_strategy == PaddingStrategy.MAX_LENGTH:
    #         if max_length is None:
    #             max_length = self.model_max_length
    #             input_ids += [self.pad_token] * (max_length - len(input_ids))
    #     return BatchEncoding(dict(input_ids=input_ids, attention_mask=[1] * len(input_ids)))
    #
    # def _batch_encode_plus(self, batch_text_or_text_pairs: Union[
    #     List[TextInput],
    #     List[TextInputPair],
    #     List[PreTokenizedInput],
    #     List[PreTokenizedInputPair],
    #     List[EncodedInput],
    #     List[EncodedInputPair],
    # ], add_special_tokens: bool = True, padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    #                        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    #                        max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False,
    #                        pad_to_multiple_of: Optional[int] = None,
    #                        return_tensors: Optional[Union[str, TensorType]] = None,
    #                        return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None,
    #                        return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False,
    #                        return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True,
    #                        **kwargs) -> BatchEncoding:
    #     raise ValueError('In LMTTokenizer._batch_encode_plus')
    #
    # def convert_tokens_to_string(self, tokens: List[str]) -> str:
    #     raise ValueError('In LMTTokenizer.convert_tokens_to_string')
    #
    # def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False,
    #             clean_up_tokenization_spaces: bool = True, **kwargs) -> str:
    #     raise ValueError('In LMTTokenizer._decode')


# class LMTTransformer(TransfoXLLMHeadModel):

def get_model_n_tokenizer(model_name: str, prec: int = 5):
    conf = TransfoXLConfig()
    ic(conf)

    if model_name == 'debug':
        n_tok = 8
        conf.update(dict(d_model=n_tok))
        model_ = TransfoXLLMHeadModel(conf)  # Initialize weights from scratch
        tokenizer_ = LMTTokenizer(prec=prec, model_max_length=n_tok)
        ic(model_)
    return tokenizer_, model_


def get_train_args(model_name: str):
    train_args = dict(
        debug=dict(
            batch_size=4,
            weight_decay=1e-2,
            lr_scheduler_type=SchedulerType.CONSTANT,
            num_train_epochs=1,
        )
    )
    bsz, decay, sch, n_ep = (train_args[model_name][k] for k in (
        'batch_size', 'weight_decay', 'lr_scheduler_type', 'num_train_epochs'
    ))
    return TrainingArguments(
        output_dir=os.path.join(DIR_DSET, DIR_MDL, model_name, now(sep='-')),
        do_train=True, do_eval=False,
        per_device_train_batch_size=bsz, per_gpu_eval_batch_size=bsz,
        learning_rate=5e-5,  # TODO: what to set?
        weight_decay=decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1,
        num_train_epochs=n_ep,
        lr_scheduler_type=sch,
        warmup_ratio=1e-2,
        log_level='warning',
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        fp16=torch.cuda.is_available(),
        fp16_full_eval=torch.cuda.is_available(),
        optim=OptimizerNames.ADAMW_TORCH,
        disable_tqdm=True,
        report_to='none',
        gradient_checkpointing=torch.cuda.is_available()
    )


def get_dataset(
        dataset_name: str,
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True
) -> datasets.Dataset:
    # TODO: only training split?
    dset = datasets.load_from_disk(os.path.join(config('path-export'), 'hf_datasets', dataset_name))
    # ic(dset[:2])
    if n_sample is not None:
        dset = dset.select(range(n_sample))
    if map_func is not None:
        num_proc = None
        n_cpu = os.cpu_count()
        if fast and n_cpu >= 2:
            num_proc = n_cpu // 2
            datasets.set_progress_bar_enabled(False)

        dset = dset.map(map_func, batched=True, remove_columns=remove_columns, num_proc=num_proc)
        datasets.set_progress_bar_enabled(True)
    dset = dset.shuffle(seed=random_seed) if random_seed is not None else dset.shuffle()
    return dset


def compute_metrics(eval_pred):
    """
    :param eval_pred: 2-tuple of (greedy prediction **ids**, labels)
        Intended to work with `CustomTrainer.prediction_step`
    """
    if not hasattr(compute_metrics, 'metric'):
        compute_metrics.metric = datasets.load_metric('accuracy')
    predictions, labels = eval_pred
    predictions = predictions.argmax(dim=-1)
    predictions, labels = predictions[:, :-1], labels[:, 1:]  # For CLM
    labels, predictions = labels.flatten(), predictions.flatten()
    msk_non_pad = (labels != PT_LOSS_PAD)
    labels, predictions = labels[msk_non_pad], predictions[msk_non_pad]
    return compute_metrics.metric.compute(predictions=predictions, references=labels)


def get_all_setup(model_name: str, dataset_name: str, prec: int = 5, n_sample=None, random_seed=None):
    tokenizer_, model_ = get_model_n_tokenizer(model_name, prec=prec)
    args = get_train_args(model_name)
    tr = get_dataset(
        dataset_name, map_func=lambda d: tokenizer_(d['text'], padding='max_length', ),
        remove_columns='title', n_sample=n_sample, random_seed=random_seed
    )
    # ic(tr.info)
    # Ensure compatibility of dataset & tokenizer, see `music_export`
    assert json.loads(tr.info.description)['precision'] == tokenizer_.prec

    trainer_ = CustomTrainer(
        model=model_, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False),
        train_dataset=tr, compute_metrics=compute_metrics
    )
    return model_, tokenizer_, tr, trainer_


class ColoredPrinterCallback(TrainerCallback):
    def __init__(self, name='LMTTransformer training'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        had_handler = False
        hd_attr_nm = 'name_for_my_logging'
        for hd in self.logger.handlers:
            if hasattr(hd, hd_attr_nm) and getattr(hd, hd_attr_nm) == name:
                had_handler = True
        if not had_handler:
            handler = logging.StreamHandler(stream=sys.stdout)  # For my own coloring
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(MyFormatter())
            setattr(handler, hd_attr_nm, name)
            self.logger.addHandler(handler)

        self.t_strt, self.t_end = None, None

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        self.logger.info(f'Training started ')
        self.t_strt = datetime.datetime.now()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.t_end = datetime.datetime.now()
        t = fmt_dt(self.t_end - self.t_strt)
        self.logger.info(f'Training completed in {logi(t)} ')

    def on_log(self, args, state, control, logs_=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(log_dict(logs_) if isinstance(logs_, dict) else logs_)


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'args' in kwargs
        self.post_init()

    def post_init(self):
        callbacks = self.callback_handler.callbacks
        # When tqdm disabled, Trainer adds a PrinterCallback, replace that with my own
        self.callback_handler.callbacks = [
            c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]
        self.add_callback(ColoredPrinterCallback())


if __name__ == '__main__':
    from icecream import ic

    # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-22 19-00-40'
    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'

    def implementation_check():
        dset = get_dataset(fnm)
        # ic(dset, dset[:2])

        tkzer = LMTTokenizer(model_max_length=12)
        ic(tkzer)
        txt = dset[0]['text']
        # txt = dset[:3]['text']
        input_ = tkzer(txt, padding='max_length', truncation=True)
        ic(input_)
        # ic(len(input_['input_ids']))
        ids_ = input_['input_ids']
        ic(tkzer.decode(ids_))
    # implementation_check()
    # exit(1)

    seed = config('random-seed')
    transformers.set_seed(seed)

    md_nm = 'debug'
    model, tokenizer, dset_tr, trainer = get_all_setup(model_name=md_nm, dataset_name=fnm)
    trainer.train()
    trainer.save_model(trainer.args.output_dir)
    # ic(model)

