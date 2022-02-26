"""
Proposed method: Transformer-XL on compact melody & bass representation for music generation
"""
import sys
from typing import Optional

import transformers
from transformers import TransfoXLConfig, TransfoXLLMHeadModel
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback
from transformers.training_args import OptimizerNames
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


def get_model_n_tokenizer(model_name: str, prec: int = 5) -> Tuple[LMTTokenizer, TransfoXLLMHeadModel]:
    conf = TransfoXLConfig()

    n_tok = None
    if model_name == 'debug':
        n_tok = 8
        conf.update(dict(d_model=n_tok))
        # ic(conf)
    model_ = MyTransfoXLLMHeadModel(conf)  # Initialize weights from scratch
    tokenizer_ = LMTTokenizer(prec=prec, model_max_length=n_tok)
    return tokenizer_, model_


def get_train_args(model_name: str) -> TrainingArguments:
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
        model_name: str, dataset_name: str, prec: int = 5, n_sample=None, random_seed=None
) -> Tuple[TransfoXLLMHeadModel, LMTTokenizer, datasets.Dataset, Trainer]:
    tokenizer_, model_ = get_model_n_tokenizer(model_name, prec=prec)
    args = get_train_args(model_name)
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


class ColoredPrinterCallback(TrainerCallback):
    def __init__(self, name='LMTTransformer training', parent_trainer: Trainer = None):
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

        self.mode = 'eval'
        self.t_strt, self.t_end = None, None

        self.trainer = parent_trainer
        args, dset_tr_, md_, tokzer = (
            getattr(parent_trainer, k) for k in ['args', 'train_dataset', 'model', 'tokenizer']
        )
        lr, n_ep = args.learning_rate, args.num_train_epochs
        self.bsz = args.per_device_train_batch_size * args.gradient_accumulation_steps
        seq_max_len = len(dset_tr_[0]['input_ids'])
        n_data, md_sz = len(dset_tr_), md_.config.d_model
        self.n_step = max(math.ceil(len(dset_tr_) // self.bsz), 1) * n_ep  # #step/epoch at least 1
        self.train_meta = OrderedDict([
            ('#data', n_data), ('model size', md_sz),
            ('learning rate', lr), ('batch shape', (self.bsz, seq_max_len)), ('#epochs', n_ep), ('#steps', self.n_step)
        ])

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        self.mode = 'train'
        self.logger.info(f'Training started with {log_dict(self.train_meta)}')
        self.t_strt = datetime.datetime.now()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.t_end = datetime.datetime.now()
        t = fmt_dt(self.t_end - self.t_strt)
        self.logger.info(f'Training completed in {logi(t)} ')
        self.mode = 'eval'

    def out_dict2str(self, d: Dict, return_wo_color: bool = False):
        keys_ = [
            'step', 'epoch', 'train_loss', 'eval_loss', 'train_acc', 'eval_acc', 'learning_rate'
        ]
        fmt = [
            f':>{len(str(self.n_step))}', ':6.2f', ':7.4f', ':7.4f', ':6.2f', ':6.2f', ':.2e'
        ]
        s_fmts = [f'{{{k}{fmt_}}}' for k, fmt_ in zip(keys_, fmt)]  # Enforce ordering

        d = {k: (
                ('loss' in k and round(v, 4)) or
                ('acc' in k and round(v * 100, 4)) or
                ('learning_rate' in k and round(v, 6)) or
                v
        ) for k, v in d.items()
        }
        s_outs = [(k, fmt_.format(**{k: d[k]})) for fmt_, k in zip(s_fmts, keys_) if k in d]
        out_ = ', '.join(f'{k}={logi(s)}' for (k, s) in s_outs)
        if return_wo_color:
            out_ = out_, ', '.join(f'{k}={s}' for (k, s) in s_outs)
        return out_

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if isinstance(logs, dict):
                if self.mode == 'train':
                    logs['step'] = state.global_step
                    logs['train_loss'] = logs.pop('loss', None)  # Trainer internal uses `loss`, instead of `train_loss`
                    logs = self.out_dict2str(logs)
                else:
                    logs = log_dict(logs)
            self.logger.info(logs)


class ClmAccCallback(ColoredPrinterCallback):
    """
    Logs training batch accuracy during CLM training

    Needs the **prediction logits** returned
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.out_dict_tr = None
        self.k_acc = 'acc_meta'

    def on_log(self, args, state, control, logs=None, **kwargs):
        def acc_stats2dict(out_dict: Dict, prefix: str) -> Dict:
            """
            Convert `acc_meta`, `classification_acc_meta` dict to stats for logging
            """
            stats_acc: pd.Series = pd.DataFrame(out_dict[self.k_acc]).sum(axis=0)
            return {f'{prefix}_acc': stats_acc.n_acc / stats_acc.n_total}

        if self.mode == 'train':
            step = state.global_step
            assert not self.trainer.do_eval  # TODO: Not supported
            if 'src' in logs and logs['src'] == 'compute_loss':
                # For gradient_accumulation, many batches of `compute_loss` may be called
                # before going into train logging
                if self.out_dict_tr is None:
                    n_ep = logs['epoch']
                    self.out_dict_tr = {'step': step, 'epoch': n_ep, self.k_acc: [logs[self.k_acc]]}
                else:  # Later batch in the same gradient accumulation
                    step_, n_ep = self.out_dict_tr['step'], self.out_dict_tr['epoch']
                    n_ep_ = logs['epoch']
                    assert step_ == step and n_ep_ == n_ep
                    self.out_dict_tr[self.k_acc].append(logs[self.k_acc])
            elif 'loss' in logs:  # The Trainer default training loss logging
                # Take the averaging by parent `Trainer` for granted
                self.out_dict_tr.update(acc_stats2dict(self.out_dict_tr, prefix='train'))
                del self.out_dict_tr[self.k_acc]
                self.out_dict_tr['learning_rate'] = logs['learning_rate']
                self.out_dict_tr['train_loss'] = logs['loss']
                self.logger.info(self.out_dict2str(self.out_dict_tr))
                self.out_dict_tr = None  # Rest for next global step
            elif any('runtime' in k for k in logs.keys()):
                self.logger.info(log_dict(logs) if isinstance(logs, dict) else logs)
            else:
                print('unhandled case', logs)
                exit(1)
        else:
            if 'src' not in logs:  # Skip custom compute_loss logging
                super().on_log(args, state, control, logs, **kwargs)


class MyTrainer(Trainer):
    def __init__(self, clm_acc_logging=True, **kwargs):
        super().__init__(**kwargs)

        self.clm_acc_logging = clm_acc_logging
        self.post_init()

    def post_init(self):
        callbacks = self.callback_handler.callbacks
        # When tqdm disabled, Trainer adds a PrinterCallback, replace that with my own
        self.callback_handler.callbacks = [
            c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]
        if self.clm_acc_logging:
            self.add_callback(ClmAccCallback(parent_trainer=self))
        else:
            self.add_callback(ColoredPrinterCallback(parent_trainer=self))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override `Trainer.compute_loss` for logging accuracy
            - Note that both training and validation calls `compute_loss`
                => Further logic needs to determine accuracy for which dataset

        Modified from https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4?u=stefanh
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Begin of added ==========================
        inputs: Dict[str, torch.Tensor]
        if self.clm_acc_logging and 'labels' in inputs:
            # ic(outputs)
            ic(outputs.logits)  # TODO
            preds = outputs.logits.detach().argmax(axis=-1)
            labels_ = inputs['labels'].detach()
            # CLM, predicting the next token given current, so shift
            preds, labels_ = preds[:, :-1], labels_[:, 1:]
            mask_non_pad = labels_ != PT_LOSS_PAD  # Consider only the actual tokens for accuracy
            preds_non_pad, labels_non_pad = preds[mask_non_pad], labels_[mask_non_pad]
            matches: torch.Tensor = (preds_non_pad == labels_non_pad)
            d_log = dict(src='compute_loss', acc_meta=dict(n_acc=matches.sum().item(), n_total=preds_non_pad.numel()))
            self.log(d_log)
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


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

    seed = config('random-seed')
    transformers.set_seed(seed)

    md_nm = 'debug'
    mdl, tokenizer, dset_tr, trainer = get_all_setup(model_name=md_nm, dataset_name=fnm)
    trainer.train()
    trainer.save_model(trainer.args.output_dir)

