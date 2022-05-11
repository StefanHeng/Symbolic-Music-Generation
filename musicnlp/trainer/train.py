"""
Proposed method: on compact melody & bass representation for autoregressive music generation
    Trying Transformer-XL, Reformer
"""
import json
import math
from os.path import join as os_join
from copy import deepcopy
from typing import List, Tuple, Dict, Union
from collections import OrderedDict

import torch
from transformers import TransfoXLConfig, ReformerModelWithLMHead, TransfoXLLMHeadModel
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer
from transformers.training_args import OptimizerNames
import datasets

from stefutil import *
from musicnlp.util import *
import musicnlp.util.train as train_util
import musicnlp.models.models as model_util
from musicnlp.vocab import MusicTokenizer
from musicnlp.preprocess import get_dataset, KeySampleDataset
from musicnlp.models import MyReformerConfig, MyTransfoXLConfig
from musicnlp.trainer import metrics


def get_model_n_tokenizer(
        model_name: str, model_size: str, prec: int = 5, model_config: Dict = None
) -> Tuple[MusicTokenizer, Union[model_util.MusicTransformerMixin, torch.nn.Module], OrderedDict]:
    assert model_name in ['xl', 'reformer'], f'Unknown model_name: {model_name}'

    tokenizer = MusicTokenizer(precision=prec)  # needed for reformer config
    if not hasattr(get_model_n_tokenizer, 'd_nm2cls'):
        get_model_n_tokenizer.d_nm2cls = {
            'xl': (MyTransfoXLConfig, TransfoXLLMHeadModel),
            'reformer': (MyReformerConfig, ReformerModelWithLMHead)
        }
    cls_config, cls_model = get_model_n_tokenizer.d_nm2cls[model_name]
    config = cls_config(model_size=model_size, tokenizer=tokenizer, **(model_config or dict()))
    # to set the correct model config for reformer, now take care of `max_length` for tokenizer
    tokenizer.model_max_length = max_length = config.model_max_length
    model_meta = OrderedDict({'model name': cls_model.__qualname__, 'max length': max_length})
    if model_name == 'reformer':
        model_meta.update(config.model_meta)
    else:
        raise NotImplementedError(f'xl')
    return tokenizer, cls_model(config=config), model_meta  # Initialize all weights from scratch


def get_train_and_my_train_args(
        model_name: str, model_size: str, train_args: Dict = None,
        my_train_args: Dict[str, Union[int, str]] = None, train_dataset: datasets.Dataset = None
) -> Tuple[TrainingArguments, Dict]:
    """
    :param model_name: Model name
    :param model_size: Model size
    :param train_args: HuggingFace Trainer args
    :param my_train_args: My own `MyTrainer` args, modifies trainer API for my customization
        - Added check pointing per k epochs
        - Logging strategy for Trainer replicated for my own console logging
        Keys trying to follow the style of Trainer
    :param train_dataset: Training dataset, intended to get size for step per epoch calculation
    """
    if not hasattr(get_train_and_my_train_args, 'default_args'):
        get_train_and_my_train_args.default_args = dict(
            output_dir=os_join(BASE_PATH, PROJ_DIR, MODEL_DIR, model_name, now(for_path=True)),
            do_train=True,
            do_eval=True,
            evaluation_strategy='epoch',
            eval_accumulation_steps=1,  # save as much GPU memory
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-08,
            max_grad_norm=1,
            warmup_ratio=1e-2,
            log_level='warning',
            logging_strategy='steps',
            logging_steps=1,
            save_strategy='epoch',
            fp16=torch.cuda.is_available(),
            optim=OptimizerNames.ADAMW_TORCH,
            disable_tqdm=True,
            report_to='none',
            # `gradient_checkpointing` for both Transformer XL and Reformer not supported
        )
    if not hasattr(get_train_and_my_train_args, 'd_train_args'):
        get_train_and_my_train_args.d_train_args = dict(
            xl={
                'debug': dict(
                    batch_size=4,
                    learning_rate=5e-4,
                    weight_decay=0,
                    lr_scheduler_type=SchedulerType.CONSTANT,
                    num_train_epochs=8,
                ),
                'debug-large': dict(
                    batch_size=8,  # To fit in colab
                    gradient_accumulation_steps=4,
                    learning_rate=5e-5,
                    weight_decay=0,
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
            },
            reformer={
                'debug': dict(
                    batch_size=8,
                    learning_rate=3e-4,
                    weight_decay=0,
                    lr_scheduler_type=SchedulerType.CONSTANT,
                    num_train_epochs=32,
                ),
                'debug-large': dict(
                    batch_size=8,
                    learning_rate=3e-4,
                    weight_decay=0,
                    lr_scheduler_type=SchedulerType.CONSTANT,
                    num_train_epochs=32,
                ),
                'tiny': dict(
                    batch_size=32,  # reformer uses batch size of 8; pop music transformer uses batch size of 16
                    learning_rate=3e-4,
                    weight_decay=1e-2,
                    lr_scheduler_type=SchedulerType.COSINE,
                    num_train_epochs=32,
                    warmup_ratio=0.1
                ),
                'small': dict(
                    batch_size=32,
                    learning_rate=3e-4,
                    weight_decay=1e-2,
                    lr_scheduler_type=SchedulerType.COSINE,
                    num_train_epochs=128,
                    warmup_ratio=0.1
                ),
                'base': dict(
                    batch_size=32,
                    learning_rate=3e-4,
                    weight_decay=1e-2,
                    lr_scheduler_type=SchedulerType.COSINE,
                    num_train_epochs=256,
                    warmup_ratio=0.1
                )
            }
        )
        d_xl = get_train_and_my_train_args.d_train_args['xl']
        for k in d_xl.keys():
            # `fp16` Doesn't work per `TransfoXL.forward`:
            # `index_copy_(): self and source expected to have the same dtype,
            # but got (self) Float and (source) Half`
            d_xl[k].update(dict(fp16=False, gradient_checkpointing=False))  # Doesn't work for `TransfoXL`
        d_ref = get_train_and_my_train_args.d_train_args['reformer']
        for k in d_ref.keys():
            d_ref[k].update(dict(gradient_checkpointing=False))  # Not supported for `Reformer`
    args = deepcopy(get_train_and_my_train_args.default_args)
    train_args_ = get_train_and_my_train_args.d_train_args[model_name][model_size]
    if 'batch_size' in train_args_:
        assert 'per_device_train_batch_size' not in train_args_ and 'per_device_eval_batch_size' not in train_args_
        bsz = train_args_.pop('batch_size')
        train_args_['per_device_train_batch_size'] = train_args_['per_device_eval_batch_size'] = bsz
    args.update(train_args_)
    if model_name == 'xl':
        assert not args['fp16'] and not args['gradient_checkpointing']
    elif model_name == 'reformer':
        assert not args['gradient_checkpointing']
    if train_args is not None:
        args.update(train_args)

    my_args: Dict[str, Union[int, str]] = dict(logging_strategy='steps', tqdm=False, augment_key=False)  # default
    if my_train_args is not None:
        my_args.update(my_train_args)
    bsz = args['per_device_train_batch_size'] * args.get('gradient_accumulation_steps', 1)
    my_args['steps_per_epoch'] = steps_per_epoch = math.ceil(len(train_dataset) / bsz)
    if 'save_epochs' in my_args:  # this is not supported by Trainer
        assert 'save_strategy' in args and args['save_strategy'] == 'epoch', \
            f'Supporting {logi("save per epoch")} error: Save strategy to Trainer should be set to {logi("epoch")}'
        args['save_strategy'] = 'steps'
        # TODO: DDP not supported
        args['save_steps'] = my_args['save_epochs'] * steps_per_epoch
    assert args['logging_strategy'] == 'steps'  # for my own internal logging to run
    assert args['disable_tqdm']  # Always use my own tqdm, see `musicnlp.util.train.MyTrainer`
    logging_strategy = my_args['logging_strategy']
    ca(log_strategy=logging_strategy)
    if logging_strategy == 'epoch':
        my_args['logging_steps'] = steps_per_epoch
    args = {k: v for k, v in args.items() if v is not None}
    return TrainingArguments(**args), my_args


class ComputeMetrics:
    def __init__(self, tokenizer: MusicTokenizer):
        self.acc = datasets.load_metric('accuracy')
        # so that no error if small #bars for now; TODO
        self.ikr = metrics.IkrMetric(tokenizer=tokenizer, n_init_bars=2)

    def __call__(self, eval_pred):
        """
        :param eval_pred: 2-tuple of (greedy prediction **ids**, labels)

        Will be the outputs on eval dataset, see `Trainer.compute_metrics`
        """
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        d_metric = dict(ikr=self.ikr(predictions, labels))

        predictions, labels = predictions[:, :-1], labels[:, 1:]  # since CLM
        labels, predictions = labels.flatten(), predictions.flatten()
        msk_non_pad = (labels != train_util.PT_LOSS_PAD)
        labels, predictions = labels[msk_non_pad], predictions[msk_non_pad]
        d_metric['ntp_acc'] = self.acc.compute(predictions=predictions, references=labels)['accuracy']
        return d_metric


def get_all_setup(
        model_name: str, model_size: str,
        dataset_names: Union[str, List[str]], prec: int = 5, n_sample=None,
        model_config: Dict = None, train_args: Dict = None, my_train_args: Dict = None
) -> Tuple[model_util.MusicTransformerMixin, MusicTokenizer, Trainer]:
    # n_sample mainly for debugging
    tokenizer, model_, meta = get_model_n_tokenizer(model_name, model_size, prec=prec, model_config=model_config)
    if my_train_args.get('augment_key', False):
        # For now, just do non-deterministic sampling for eval set too, TODO?
        dset = KeySampleDataset.from_hf(dataset_names, tokenizer=tokenizer, get_dataset_kwargs=dict(n_sample=n_sample))
    else:
        dset = get_dataset(
            dataset_names=dataset_names,
            map_func=lambda d: tokenizer(d['score'], padding='max_length', truncation=True),
            remove_columns=['title', 'score', 'duration'], n_sample=n_sample  # i.e. keep the input ids only
        )
    tr, vl = dset['train'], dset['test']
    args, my_args, = get_train_and_my_train_args(model_name, model_size, train_args, my_train_args, tr)
    assert all(  # Ensure compatibility of dataset & tokenizer, see `music_export`
        get(json.loads(ds.info.description), 'extractor_meta.precision') == tokenizer.precision for ds in dset.values()
    )

    clm_acc_logging = isinstance(model_, ReformerModelWithLMHead)  # couldn't get logits for `TransfoXL`
    cm = ComputeMetrics(tokenizer)
    trainer_ = train_util.MyTrainer(
        model_meta=meta,
        clm_acc_logging=clm_acc_logging, my_args=my_args,
        train_metrics=dict(ikr=cm.ikr),
        model=model_, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=tr, eval_dataset=vl, compute_metrics=cm
    )
    return model_, tokenizer, trainer_


if __name__ == '__main__':
    import transformers
    from icecream import ic

    ic.lineWrapWidth = 400

    def check_model_size():
        md_nm = 'reformer'
        # md_sz = 'small'
        # md_sz = 'base'
        md_sz = 'large'
        mdl: torch.nn.Module = get_model_n_tokenizer(model_name=md_nm, model_size=md_sz)[1]
        ic(get_model_num_trainable_parameter(mdl))
    # check_model_size()

    def train(resume_from_checkpoint: str = None):
        seed = sconfig('random-seed')

        md_nm = 'reformer'
        md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        # md_sz = 'small'
        # md_sz = 'base'
        ic(md_nm, md_sz)

        if md_nm != 'reformer':
            transformers.set_seed(seed)
        # not set seed if reformer for LSH attention,
        # see https://huggingface.co/docs/transformers/model_doc/reformer#transformers.ReformerConfig.hash_seed

        augment_key = True
        if augment_key:
            dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, ' \
                     'meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
            dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
                      'meta={mode=melody, prec=5, th=1}, 2022-04-17_11-52-15'
        else:
            dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, ' \
                      'meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
            dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
                      'meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52'
        dnms = [dnm_909, dnm_lmd]

        if 'debug' in md_sz or md_sz == 'tiny':
            # n = None
            n = 8
            train_args = dict(
                per_device_train_batch_size=2,
                # save_strategy='no',
                save_strategy='epoch',
                num_train_epochs=16,
            )
            my_train_args = dict(
                save_epochs=16,
                # logging_strategy='no',
                # logging_strategy='steps',
                logging_strategy='epoch',
                tqdm='train-only',
                augment_key=augment_key,
            )
        else:
            n = None
            train_args = dict(num_train_epochs=16)
            my_train_args = dict(
                logging_strategy='epoch',
                save_epochs=4
            )
        mdl, tokenizer, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, dataset_names=dnms, n_sample=n,
            train_args=train_args, my_train_args=my_train_args
        )

        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint)
        else:
            trainer.train()
        trainer.save_model(os_join(trainer.args.output_dir, 'trained'))
    train()

    # checkpoint_path = os_join(PATH_BASE, DIR_PROJ, DIR_MDL, 'reformer', '2022-04-03_00-20-53', 'checkpoint-1856')
    # train(resume_from_checkpoint=checkpoint_path)
