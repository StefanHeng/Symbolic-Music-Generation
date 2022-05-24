import os
import re
import math
from os.path import join as os_join
from typing import List, Tuple, Dict, Callable, Any, Optional, Union
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from stefutil import *


PT_LOSS_PAD = -100  # Pytorch indicator value for ignoring loss, used in huggingface for padding tokens


def meta2fnm_meta(meta: Dict) -> Dict:
    if not hasattr(meta2fnm_meta, 'd_key'):
        meta2fnm_meta.d_key = {
            'model name': 'nm', 'max length': 'l', 'axial_pos_shape': 'ax_pos_sp',
            'hidden_size': 'hd_sz', 'ff_size': 'ff_sz',
            'n_layer': 'n_l', 'attn_layers': 'attn', 'attention_shape': 'attn_sh',
            'parameter_count': 'n_param'
        }
    return OrderedDict((meta2fnm_meta.d_key[k_], v) for k_, v in meta.items())


class MyEvalPrediction:
    """
    Override to pass in `key_scores`

    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
        key_scores (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        key_scores: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.key_scores = key_scores

    def __iter__(self):
        # if self.inputs is not None:
        #     return iter((self.predictions, self.label_ids, self.inputs))
        # else:
        #     return iter((self.predictions, self.label_ids))
        return iter(e for e in (self.predictions, self.label_ids, self.inputs, self.key_scores) if e is not None)

    def __getitem__(self, idx):
        # if idx < 0 or idx > 2:
        #     raise IndexError("tuple index out of range")
        if idx < 0 or idx > 3:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 3 and self.key_scores is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
        elif idx == 3:
            return self.key_scores


class MyTrainer(Trainer):
    def __init__(
            self, model_meta: Dict = None, my_args: Dict = None,
            clm_acc_logging=True, train_metrics: Dict[str, Callable] = None,
            compute_metrics: Optional[Callable[[MyEvalPrediction], Dict]] = None,
            **kwargs
    ):
        assert kwargs['args'].disable_tqdm  # Always disable
        kwargs['compute_metrics'] = compute_metrics
        super().__init__(**kwargs)
        self.clm_acc_logging = clm_acc_logging
        self.model_meta = model_meta
        self.model_meta['parameter_count'] = get_model_num_trainable_parameter(self.model)
        self.name = self.model.__class__.__qualname__
        self.train_metrics = train_metrics

        self.my_args = my_args
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
        callback_cls = ColoredPrinterCallbackForClm if self.clm_acc_logging else ColoredPrinterCallback
        if not self.clm_acc_logging:
            raise NotImplementedError('on-CLM task logging not updated')
        if self.my_args['tqdm']:
            self.add_callback(MyProgressCallback(train_only=self.my_args['tqdm'] == 'train-only'))
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
        if self.is_in_train and self.clm_acc_logging and 'labels' in inputs:
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
            ntp_acc_meta = dict(matched=matches.sum().item(), total=preds_non_pad.numel())
            d_log['ntp_acc_meta'] = ntp_acc_meta
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

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        # ========================== Begin of added =========================
        from transformers.file_utils import is_sagemaker_mp_enabled
        from transformers.trainer_pt_utils import nested_detach
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
        # ========================== End of added =========================
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        # ========================== Begin of modified =========================
        return (loss, logits, labels, inputs['key_scores'].detach())
        # return (loss, logits, labels)
        # ========================== End of modified =========================

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Override for passing in `key_scores`, `include_inputs_for_metrics` only returns `input_ids` so not helpful
        """
        # ========================== Begin of added =========================
        from transformers.trainer_utils import has_length
        from transformers.file_utils import is_torch_tpu_available
        if is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
        from transformers.deepspeed import deepspeed_init
        from transformers.trainer_utils import denumpify_detensorize
        from transformers.trainer_pt_utils import (
            find_batch_size, nested_concat, nested_numpify, nested_truncate, IterableDatasetShard
        )
        # ========================== End of added =========================
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        # ========================== Begin of added =========================
        from transformers.utils import logging
        logger = logging.get_logger(__name__)
        # ========================== End of added =========================
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        # ========================== Begin of added =========================
        key_scores_host = None
        # ========================== End of added =========================
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # ========================== Begin of added =========================
        all_key_scores = None
        # ========================== End of added =========================
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            # ========================== Begin of modified =========================
            # loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            loss, logits, labels, key_scores = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            # ========================== Begin of modified =========================
            inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            # ========================== Begin of added =========================
            if key_scores is not None:
                key_scores = self._pad_across_processes(key_scores)
                key_scores = self._nested_gather(key_scores)
                key_scores_host = (
                    key_scores if key_scores_host is None
                    else nested_concat(key_scores_host, key_scores, padding_index=-100)
                )
            # ========================== End of added =========================
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # ========================== Begin of added =========================
                if key_scores_host is not None:
                    key_scores = nested_numpify(key_scores_host)
                    all_key_scores = (
                        key_scores if all_key_scores is None
                        else nested_concat(all_key_scores, key_scores, padding_index=-100)
                    )
                # ========================== End of added =========================

                # Set back to None to begin a new accumulation
                # ========================== Begin of modified =========================
                # losses_host, preds_host, labels_host, inputs_host = None, None, None, None
                losses_host, preds_host, inputs_host, labels_host, key_scores_host = None, None, None, None, None
                # ========================== End of modified =========================

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        # ========================== Begin of added =========================
        if key_scores_host is not None:
            key_scores = nested_numpify(key_scores_host)
            all_key_scores = (
                key_scores if key_scores_host is None
                else nested_concat(key_scores, key_scores_host, padding_index=-100)
            )
        # ========================== End of added =========================

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        # ========================== Begin of added =========================
        if all_key_scores is not None:
            all_key_scores = nested_truncate(all_key_scores, num_samples)
        # ========================== End of added =========================

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            # ========================== Begin of modified =========================
            # if args.include_inputs_for_metrics:
            #     metrics = self.compute_metrics(
            #         EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
            #     )
            # else:
            #     metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(MyEvalPrediction(
                    predictions=all_preds, label_ids=all_labels, inputs=all_inputs, key_scores=all_key_scores
                ))
            else:
                metrics = self.compute_metrics(MyEvalPrediction(
                    predictions=all_preds, label_ids=all_labels, key_scores=all_key_scores
                ))
            # ========================== End of modified =========================
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


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
        self.log_fnm = f'md={log_dict_p(meta)}, n={n_data}, a={lr}, bsz={self.bsz}, n_ep={n_ep}'

        if name is None:
            name = 'MyTrainer'
        self.name = f'{name} Training'
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
            self.writer = SummaryWriter(os_join(self.output_dir, f'tb - {self.log_fnm}'))

        conf = self.trainer.model.config.to_dict()
        train_args = self.trainer.args.to_dict()
        meta = self.trainer.model_meta
        self.logger.info(f'Training started with model {log_dict(meta)}, {log_dict_pg(conf)} '
                         f'on {log_dict(self.train_meta)} with training args {log_dict_pg(train_args)} '
                         f'and my training args {log_dict(self.trainer.my_args)}... ')
        self.logger_fl.info(f'Training started with with model {log_dict_nc(meta)}, {log_dict_id(conf)} '
                            f'on {log_dict_nc(self.train_meta)} with training args {log_dict_id(train_args)} '
                            f'and my training args {log_dict_nc(self.trainer.my_args)}... ')
        self.t_strt = datetime.datetime.now()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.t_end = datetime.datetime.now()
        t = fmt_delta(self.t_end - self.t_strt)
        self.logger.info(f'Training completed in {logi(t)} ')
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
                    self.logger.info(log_dict(logs))
                    self.logger_fl.info(log_dict_nc(logs))
                else:
                    self.logger.info(log_dict(logs))
                    self.logger_fl.info(log_dict(logs, with_color=False))
            else:
                self.logger.info(logs)
                self.logger_fl.info(logs)


no_prefix = ['epoch', 'step']


def add_prefix(key: str) -> bool:
    return key not in no_prefix


class ColoredPrinterCallbackForClm(ColoredPrinterCallback):
    """
    Additionally log next-token-prediction accuracy

    .. note:: Music-theory-inspired in-key-ratio also added, see `models.metrics.py`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_dict = None  # For passing NTP accuracy in training steps
        self.pattern_eval_key = re.compile(r'^eval_(?P<key>.*)$')

    def _get_eval_key(self, key: str) -> str:
        return self.pattern_eval_key.match(key).group('key')

    def _log(self, d_log, mode='train', to_console=True):
        ca(log_mode=mode)
        d_log_write = {f'{mode}/{k}' if add_prefix(k) else k: v for k, v in d_log.items()}
        d_log_write = self.prettier(d_log_write)
        if to_console:
            self.logger.info(log_dict(d_log_write))
        self.logger_fl.info(log_dict_nc(d_log_write))

        step = d_log.get('step') if mode == 'train' else d_log.get('epoch')
        for k, v in d_log.items():
            if add_prefix(k):
                self.writer.add_scalar(tag=f'{mode}/{k}', scalar_value=v, global_step=step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'src' in logs and logs['src'] == 'compute_loss':
                del logs['src']
                self.out_dict = logs
            else:
                # Heuristics on the training step updates, see `Trainer._maybe_log_save_evaluate`
                if self.mode == 'train' and all('runtime' not in k for k in logs):
                    step = state.global_step
                    assert logs['epoch'] == round(state.epoch, 2)
                    n_ep = state.epoch  # The one originally is rounded, see `Trainer.log`
                    loss, lr = logs['loss'], logs['learning_rate']
                    assert self.out_dict is not None
                    ntp_acc_meta = self.out_dict['ntp_acc_meta']
                    # TODO: Potentially support gradient accumulation
                    # ntp_acc_meta = {k: sum(v for v in d[k]) for k, d in ntp_acc_meta.items()}
                    ntp_acc = ntp_acc_meta['matched'] / ntp_acc_meta['total']
                    metrics = {k: self.out_dict[k] for k in self.trainer.train_metrics}

                    self.out_dict = OrderedDict([
                        ('step', step), ('epoch', n_ep), ('learning_rate', lr),
                        ('loss', loss), ('ntp_acc', ntp_acc)
                    ])
                    self.out_dict.update(metrics)

                    # `should_log` in Trainer just prevents the `on_log` call, I only filter console logging
                    should_log = False
                    my_log_strat = self.trainer.my_args.get('logging_strategy', 'steps')
                    if my_log_strat == 'steps':
                        should_log = True
                    elif my_log_strat == 'epoch' and step % self.trainer.my_args['steps_per_epoch'] == 0:
                        should_log = True
                    self._log(self.out_dict, mode='train', to_console=should_log)
                elif 'eval_loss' in logs:  # `Trainer.is_in_train` is not False so can't use
                    assert 'epoch' in logs
                    del logs['epoch']
                    # For potential metrics computed
                    ks = [self._get_eval_key(k) for k in logs.keys()]
                    ks = [
                        k for k in ks
                        if k not in ['loss', 'ntp_acc', 'runtime', 'samples_per_second', 'steps_per_second']
                    ]
                    ks = ['loss', 'ntp_acc'] + ks
                    # Log eval on an epoch-level, always logged irrelevant to `logging_strategy`
                    # Evaluation finished during training; TODO: didn't verify other positive cases
                    n_ep = state.epoch
                    assert n_ep.is_integer()
                    d_log = dict(step=state.global_step, epoch=int(n_ep), **{k: logs[f'eval_{k}'] for k in ks})
                    self._log(d_log, mode='eval', to_console=True)
                else:
                    self.logger.info(log_dict(logs))
                    self.logger_fl.info(log_dict_nc(logs))


class ClmAccCallback(ColoredPrinterCallback):
    """
    Logs training batch accuracy during CLM training

    Needs the **prediction logits** returned
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.out_dict_tr = None
        self.k_acc = 'ntp_acc'

    def on_log(self, args, state, control, logs=None, **kwargs):
        def acc_stats2dict(out_dict: Dict, prefix: str) -> Dict:
            """
            Convert `acc_meta`, `classification_acc_meta` dict to stats for logging
            """
            stats_acc: pd.Series = pd.DataFrame(out_dict[self.k_acc]).sum(axis=0)
            return {f'{prefix}_acc': stats_acc.n_acc / stats_acc.n_total}

        if self.mode == 'train':
            step = state.global_step
            assert not self.trainer.args.do_eval  # TODO: Not supported
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
                self.logger.info(self.prettier(self.out_dict_tr))
                self.out_dict_tr = None  # Rest for next global step
            elif any('runtime' in k for k in logs.keys()):
                self.logger.info(log_dict(logs) if isinstance(logs, dict) else logs)
            else:
                print('unhandled case', logs)
                exit(1)
        else:
            if 'src' not in logs:  # Skip custom compute_loss logging
                super().on_log(args, state, control, logs, **kwargs)
