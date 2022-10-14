import os
import re
import math
from os.path import join as os_join
from typing import List, Dict, Callable
import datetime
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from stefutil import *


__all__ = [
    'PT_LOSS_PAD', 'MyTrainer',
    'ColoredPrinterCallback', 'ColoredPrinterCallbackForClm'
]

PT_LOSS_PAD = -100  # Pytorch indicator value for ignoring loss, used in huggingface for padding tokens


def meta2fnm_meta(meta: Dict, subset: str = ('model name', 'max length', 'hidden_size', 'attention_shape')) -> Dict:
    if not hasattr(meta2fnm_meta, 'd_key'):
        meta2fnm_meta.d_key = {
            'model name': 'nm', 'max length': 'l', 'axial_pos_shape': 'ax_pos_sp',
            'hidden_size': 'hd_sz', 'ff_size': 'ff_sz',
            'n_layer': 'n_l', 'attn_layers': 'attn', 'attention_shape': 'attn_sh',
            'parameter_count': 'n_param', 'seg_len': 'seg_len', 'max_len': 'max_len'
        }
    if subset:
        meta = {k: v for k, v in meta.items() if k in subset}
    return OrderedDict((meta2fnm_meta.d_key[k_], v) for k_, v in meta.items())


class MyTrainer(Trainer):
    def __init__(
            self, model_meta: Dict = None, my_args: Dict = None,
            monitor_ntp_acc=True, train_metrics: Dict[str, Callable] = None,
            disable_train_metrics=False,
            **kwargs
    ):
        assert kwargs['args'].disable_tqdm  # Always disable
        super().__init__(**kwargs)
        self.monitor_ntp_acc = monitor_ntp_acc
        self.model_meta = model_meta
        self.name = model_meta.get('model name', self.model.__class__.__qualname__)
        self.train_metrics = train_metrics
        self.disable_train_metrics = disable_train_metrics

        self.my_args = my_args
        self.with_tqdm = my_args.get('tqdm', False)
        self.post_init()

    def post_init(self):
        callbacks = self.callback_handler.callbacks
        # Trainer adds a `PrinterCallback` or a `ProgressCallback`, replace all that with my own,
        # see `MyProgressCallback`
        self.callback_handler.callbacks = [
            c for c in callbacks if str(c.__class__) not in [
                "<class 'transformers.trainer_callback.ProgressCallback'>",
                "<class 'transformers.trainer_callback.PrinterCallback'>"
            ]
        ]
        callback_cls = ColoredPrinterCallbackForClm if self.monitor_ntp_acc else ColoredPrinterCallback
        if not self.monitor_ntp_acc:
            raise NotImplementedError('on-CLM task logging not updated')
        if self.with_tqdm:
            self.add_callback(MyProgressCallback(train_only=self.with_tqdm == 'train-only'))
        self.add_callback(callback_cls(name=self.name, parent_trainer=self))

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
        # don't need to calculate NTP acc here for eval, see `compute_ntp_acc`
        if model.training and self.monitor_ntp_acc and 'labels' in inputs and (not self.disable_train_metrics):
            preds = outputs.logits.detach().argmax(axis=-1)
            labels_ = inputs['labels'].detach()
            d_log = dict(src='compute_loss')
            if self.train_metrics:
                ks = None
                if 'key_scores' in inputs:
                    ks = inputs['key_scores'].detach()
                d_log.update({k: f(preds, labels_, ks) for k, f in self.train_metrics.items()})

            # CLM, predicting the next token given current, so shift
            preds, labels_ = preds[:, :-1], labels_[:, 1:]
            msk_not_pad = labels_ != PT_LOSS_PAD  # Consider only the actual tokens for accuracy
            preds_non_pad, labels_non_pad = preds[msk_not_pad], labels_[msk_not_pad]
            matches: torch.Tensor = (preds_non_pad == labels_non_pad)
            # next-token-prediction task
            d_log['ntp_acc'] = matches.sum().item() / preds_non_pad.numel()
            self.log(d_log)
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class ColoredPrinterCallback(TrainerCallback):
    """
    Supports colored terminal output, logging file write, data sent to tensorboard for plotting

    Evaluation during training **not supported**
    """
    def __init__(self, name: str = None, parent_trainer: MyTrainer = None, report2tb: bool = True):
        self.mode = 'eval'
        self.t_strt, self.t_end = None, None

        self.trainer = parent_trainer
        args, dset_tr_, md_, tokzer = (
            getattr(parent_trainer, k) for k in ['args', 'train_dataset', 'model', 'tokenizer']
        )
        lr, n_ep = args.learning_rate, args.num_train_epochs
        self.bsz = args.per_device_train_batch_size * args.gradient_accumulation_steps
        seq_max_len = len(dset_tr_[0]['input_ids'])
        n_data = len(dset_tr_)
        self.n_step = max(math.ceil(len(dset_tr_) / self.bsz), 1) * n_ep  # #step/epoch at least 1
        self.train_meta = OrderedDict([
            ('#data', n_data), ('batch shape', (self.bsz, seq_max_len)),
            ('#epochs', n_ep), ('#steps', self.n_step), ('learning rate', lr),
        ])

        self.output_dir = self.trainer.args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_time = self.output_dir.split(os.sep)[-1]  # expect last dir name as time stamp
        meta = meta2fnm_meta(self.trainer.model_meta)
        self.log_fnm = f'Train_{pl.pa(meta)}_{{n={n_data}, a={lr}, bsz={self.bsz}, n_ep={n_ep}}}'

        name = name or 'MyTrainer'
        self.name = f'{name} Train'
        self.logger, self.logger_fl, self.writer = None, None, None
        self.report2tb = report2tb
        self.prettier = MlPrettier(ref=self.train_meta, metric_keys=['acc', 'recall', 'auc', 'ikr'])

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        self.mode = 'train'

        self.logger = get_logger(self.name)
        self.logger_fl = get_logger(
            name=self.name, typ='file-write', file_path=os_join(self.output_dir, f'{self.log_fnm}.log')
        )
        if self.report2tb:
            self.writer = SummaryWriter(os_join(self.output_dir, f'TB_{self.log_fnm}'))

        conf = self.trainer.model.config.to_dict()
        train_args = self.trainer.args.to_dict()
        meta = self.trainer.model_meta
        self.logger.info(f'Training started with model {pl.i(meta)}, {pl.fmt(conf)} '
                         f'on {pl.i(self.train_meta)} with training args {pl.fmt(train_args)} '
                         f'and my training args {pl.i(self.trainer.my_args)}... ')
        self.logger_fl.info(f'Training started with with model {pl.nc(meta)}, {pl.id(conf)} '
                            f'on {pl.nc(self.train_meta)} with training args {pl.id(train_args)} '
                            f'and my training args {pl.nc(self.trainer.my_args)}... ')
        self.logger.info(f'Logging will be saved to {pl.i(self.log_fnm)}... ')
        self.t_strt = datetime.datetime.now()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.t_end = datetime.datetime.now()
        t = fmt_delta(self.t_end - self.t_strt)
        self.logger.info(f'Training completed in {pl.i(t)} ')
        self.logger_fl.info(f'Training completed in {t} ')
        self.mode = 'eval'

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if isinstance(logs, dict):
                # Heuristics on the training step updates, see `Trainer._maybe_log_save_evaluate`
                if self.mode == 'train' and all('runtime' not in k for k in logs):
                    logs['step'] = step = state.global_step
                    assert logs['epoch'] == round(state.epoch, 2)
                    logs['epoch'] = state.epoch  # The one originally is rounded, see `Trainer.log`
                    # Trainer internal uses `loss`, instead of `train_loss`
                    logs['train_loss'] = loss = logs.pop('loss', None)
                    assert loss is not None
                    lr = logs['learning_rate']
                    self.writer.add_scalar('Train/loss', loss, step)
                    self.writer.add_scalar('Train/learning_rate', lr, step)

                    # out_console, out_write = self.out_dict2str(logs, return_wo_color=True)  # TODO: didn't test
                    logs = self.prettier(logs)
                    self.logger.info(pl.i(logs))
                    self.logger_fl.info(pl.nc(logs))
                else:
                    self.logger.info(pl.i(logs))
                    self.logger_fl.info(pl.i(logs, with_color=False))
            else:
                self.logger.info(logs)
                self.logger_fl.info(logs)


class ColoredPrinterCallbackForClm(ColoredPrinterCallback):
    """
    Additionally log next-token-prediction accuracy

    .. note:: Music-theory-inspired in-key-ratio also added, see `models.metrics.py`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # For passing metrics including NTP accuracy in training steps
        self._train_step_metrics: List[Dict[str, float]] = []
        self.gas = self.trainer.args.gradient_accumulation_steps
        self.pattern_eval_key = re.compile(r'^eval_(?P<key>.*)$')

        self.ls = None

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.ls = LogStep(
            trainer=self.trainer, prettier=self.prettier,
            logger=self.logger, file_logger=self.logger_fl, tb_writer=self.writer
        )

    def _get_eval_key(self, key: str) -> str:
        return self.pattern_eval_key.match(key).group('key')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            step, training = state.global_step, self.mode == 'train'
            if 'src' in logs and logs['src'] == 'compute_loss':
                del logs['src']
                del logs['epoch']
                mic(logs)
                self._train_step_metrics.append(logs)  # the only metric needed for gathering
            elif training and all('runtime' not in k for k in logs):  # Training step
                loss, lr = logs['loss'], logs['learning_rate']

                d_log = OrderedDict(dict(step=step, epoch=state.epoch, learning_rate=lr, loss=loss))
                if not self.trainer.disable_train_metrics:
                    assert len(self._train_step_metrics) == self.gas
                    metrics = {
                        k: sum(d[k] for d in self._train_step_metrics) / self.gas
                        for k in self._train_step_metrics[0].keys()
                    }
                    mic(metrics)
                    self._train_step_metrics = []  # for next iter
                    d_log['ntp_acc'] = metrics.pop('ntp_acc')  # always log NTP acc first
                    d_log.update(metrics.items())

                # `should_log` in Trainer just prevents the `on_log` call, I only filter console logging
                should_log = False
                my_log_strat = self.trainer.my_args.get('logging_strategy', 'steps')
                if my_log_strat == 'steps':
                    should_log = True
                elif my_log_strat == 'epoch' and step % self.trainer.my_args['steps_per_epoch'] == 0:
                    should_log = True
                self.ls(d_log, training=training, to_console=should_log)
            elif 'eval_loss' in logs:  # `Trainer.is_in_train` is not False so can't use
                # Get all potential metrics computed
                ks = [self._get_eval_key(k) for k in logs.keys()]
                custom_keys = ['loss', 'ntp_acc', 'runtime', 'samples_per_second', 'steps_per_second']
                default_keys = [k for k in ks if k not in custom_keys]
                ks = ['loss', 'ntp_acc'] + default_keys

                d_log = dict(step=state.global_step, epoch=int(state.epoch), **{k: logs[f'eval_{k}'] for k in ks})
                should_log = self.trainer.my_args.get('logging_strategy', 'steps') != 'no'
                self.ls(d_log, training=training, to_console=should_log)

                # for next iter, cos `compute_loss` is called for eval too
                # we don't need it for logging as eval metrics are gathered by Trainer out-of-the-box
                self._train_step_metrics = []
            else:
                self.logger.info(pl.i(logs))
                self.logger_fl.info(pl.nc(logs))
