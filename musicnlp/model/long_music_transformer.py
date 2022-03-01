"""
Proposed method: Transformer-XL on compact melody & bass representation for music generation
"""
from typing import Optional

import torch
import datasets
from tokenizers import AddedToken
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer
from transformers.training_args import OptimizerNames

from musicnlp.util import *
from musicnlp.util.train import PT_LOSS_PAD, MyTrainer


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
        return text.split()  # Each word in vocab is split by space; TODO: special token handling?

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.spec_toks_enc)

    def _convert_token_to_id(self, token):
        return self.spec_toks_enc[token] if token in self.spec_toks_enc else self.vocab.t2i(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.spec_toks_dec[index] if index in self.spec_toks_dec else self.vocab.i2t(index)


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


class MyTransfoXLLMHeadModel(TransfoXLLMHeadModel):
    """
    For compatibility with huggingface Trainer API
    See https://github.com/huggingface/transformers/issues/11822#issuecomment-847614016
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ic(self.config)
        # exit(1)

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


def get_model_n_tokenizer(
        model_name: str, prec: int = 5, model_config: Dict = None
) -> Tuple[LMTTokenizer, TransfoXLLMHeadModel]:
    conf = TransfoXLConfig()

    if 'debug' in model_name:
        n_tok = 512 if 'large' in model_name else 8
    else:
        assert 'small' in model_name
        n_tok = 1024
    conf.update(dict(d_model=n_tok))
    if model_config is not None:
        conf.update(model_config)
    model_ = MyTransfoXLLMHeadModel(conf)  # Initialize weights from scratch
    tokenizer_ = LMTTokenizer(prec=prec, model_max_length=model_.config.d_model)
    return tokenizer_, model_


def get_train_args(model_name: str, train_args: Dict = None) -> TrainingArguments:
    train_args_ = {
        'debug': dict(
            batch_size=4,
            learning_rate=5e-4,
            weight_decay=1e-2,
            lr_scheduler_type=SchedulerType.CONSTANT,
            num_train_epochs=8,
        ),
        'debug-large': dict(
            batch_size=8,  # To fit in colab
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=1e-2,
            lr_scheduler_type=SchedulerType.CONSTANT,
            num_train_epochs=3
        ),
        'small': dict(
            batch_size=32,
            learning_rate=4e-5,
            weight_decay=1e-2,
            lr_scheduler_type=SchedulerType.COSINE,
            num_train_epochs=32
        )
    }
    bsz, lr, decay, sch, n_ep, gas = (train_args_[model_name].get(k, None) for k in (
        'batch_size', 'learning_rate', 'weight_decay',
        'lr_scheduler_type', 'num_train_epochs', 'gradient_accumulation_steps'
    ))
    args = dict(
        output_dir=os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, now(sep='-')),
        do_train=True, do_eval=False,
        per_device_train_batch_size=bsz, per_gpu_eval_batch_size=bsz,
        gradient_accumulation_steps=gas,
        learning_rate=lr,  # TODO: what to set?
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
        # fp16=torch.cuda.is_available(),
        # Doesn't work per `TransfoXL.forward`:
        # `index_copy_(): self and source expected to have the same dtype, but got (self) Float and (source) Half`
        fp16=False,
        optim=OptimizerNames.ADAMW_TORCH,
        disable_tqdm=True,
        report_to='none',
        # gradient_checkpointing=torch.cuda.is_available()
        gradient_checkpointing=False  # Doesn't work for `TransfoXL`
    )
    if train_args is not None:
        args.update(train_args)
    args = {k: v for k, v in args.items() if v is not None}
    return TrainingArguments(**args)


def get_dataset(
        dataset_name: str,
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True
) -> datasets.Dataset:
    # TODO: only training split?
    dset = datasets.load_from_disk(os.path.join(config('path-export'), 'hf_datasets', dataset_name))
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


def get_all_setup(
        model_name: str, dataset_name: str, prec: int = 5, n_sample=None, random_seed=None,
        model_config: Dict = None, train_args: Dict = None
) -> Tuple[TransfoXLLMHeadModel, LMTTokenizer, datasets.Dataset, Trainer]:
    tokenizer_, model_ = get_model_n_tokenizer(model_name, prec=prec, model_config=model_config)
    args = get_train_args(model_name, train_args)
    tr = get_dataset(
        dataset_name, map_func=lambda d: tokenizer_(d['text'], padding='max_length', truncation=True),
        remove_columns=['title', 'text'], n_sample=n_sample, random_seed=random_seed
    )
    # Ensure compatibility of dataset & tokenizer, see `music_export`
    assert json.loads(tr.info.description)['precision'] == tokenizer_.prec

    trainer_ = MyTrainer(
        clm_acc_logging=False,  # TODO: get logits for transformer-xl?
        model=model_, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False),
        train_dataset=tr, compute_metrics=compute_metrics
    )
    return model_, tokenizer_, tr, trainer_


if __name__ == '__main__':
    import transformers
    from icecream import ic

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'

    def implementation_check():
        dset = get_dataset(fnm)
        # ic(dset, dset[:2])

        tkzer = LMTTokenizer(model_max_length=12)
        ic(tkzer, tkzer.model_max_length)
        txt = dset[1]['text']
        # txt = dset[:3]['text']
        # Turning off both `padding` & `truncation`, and the token ids too long warning appears
        input_ = tkzer(txt, padding='max_length', truncation=True)
        ic(input_)
        # ic(len(input_['input_ids']))
        ids_ = input_['input_ids']
        ic(tkzer.decode(ids_))
    # implementation_check()

    def train():
        seed = config('random-seed')
        transformers.set_seed(seed)

        md_nm = 'debug'
        # md_nm = 'debug-large'

        # n = 4
        n = None

        mdl, tokenizer, dset_tr, trainer = get_all_setup(model_name=md_nm, dataset_name=fnm, n_sample=n, random_seed=seed)
        trainer.train()
        trainer.save_model(os.path.join(trainer.args.output_dir, 'final-trained'))
    # train()
