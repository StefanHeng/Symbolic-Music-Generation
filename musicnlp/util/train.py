import sys

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback

from .util import *


PT_LOSS_PAD = -100  # Pytorch indicator value for ignoring loss, used in huggingface for padding tokens


class ColoredPrinterCallback(TrainerCallback):
    def __init__(self, name='LMTTransformer training', parent_trainer: Trainer = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        had_handler = False
        hd_attr_nm = 'name_for_my_logging'
        for hd in self.logger.handlers:
            if hasattr(hd, hd_attr_nm) and getattr(hd, hd_attr_nm) == name:
                had_handler = True
        if not had_handler:  # For ipython compatibility
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
