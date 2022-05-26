"""
Proposed method: on compact melody & bass representation for autoregressive music generation
    Trying Transformer-XL, Reformer
"""
import json
import math
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Optional
from collections import OrderedDict

import torch
from transformers import ReformerModelWithLMHead
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer
from transformers.training_args import OptimizerNames
import datasets

from stefutil import *
from musicnlp.util import *
import musicnlp.util.train as train_util
from musicnlp.vocab import MusicTokenizer, key_ordinal2str
from musicnlp.preprocess import get_dataset, KeySampleDataset
from musicnlp.models import MyReformerConfig, MyTransfoXLConfig, MyTransfoXLLMHeadModel
from musicnlp.trainer import metrics


def get_model_n_tokenizer(
        model_name: str, model_size: str, prec: int = 5, model_config: Dict = None
) -> Tuple[MusicTokenizer, torch.nn.Module, OrderedDict]:
    ca.check_mismatch('Model Name', model_name, ['transf-xl', 'reformer'])

    tokenizer = MusicTokenizer(precision=prec)  # needed for reformer config
    if not hasattr(get_model_n_tokenizer, 'd_nm2cls'):
        get_model_n_tokenizer.d_nm2cls = {
            'transf-xl': (MyTransfoXLConfig, MyTransfoXLLMHeadModel),
            'reformer': (MyReformerConfig, ReformerModelWithLMHead)
        }
    cls_config, cls_model = get_model_n_tokenizer.d_nm2cls[model_name]
    config = cls_config(model_size=model_size, tokenizer=tokenizer, **(model_config or dict()))
    # to set the correct model config for reformer, now take care of `max_length` for tokenizer
    tokenizer.model_max_length = max_length = config.max_length_
    model_meta = OrderedDict({'model name': cls_model.__qualname__, 'max length': max_length})
    model_meta.update(config.model_meta)
    return tokenizer, cls_model(config=config), model_meta  # Initialize all weights from scratch


class TrainArgs:
    model_name2preset = {
        'transf-xl': {
            'debug': dict(
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0,
                lr_scheduler_type=SchedulerType.CONSTANT,
                num_train_epochs=64,
            ),
            'debug-large': dict(
                batch_size=8,
                learning_rate=1e-3,
                weight_decay=0,
                lr_scheduler_type=SchedulerType.CONSTANT,
                num_train_epochs=16
            ),
            'tiny': dict(
                batch_size=32,
                learning_rate=3e-4,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                num_train_epochs=64
            ),
            'small': dict(
                batch_size=32,
                learning_rate=3e-4,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                num_train_epochs=64
            ),
            'base': dict(
                batch_size=32,
                learning_rate=3e-4,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                num_train_epochs=64
            )
        },
        'reformer': {
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
    }

    def __init__(self, model_name: str, model_size: str):
        self.model_name, self.model_size = model_name, model_size

    @staticmethod
    def _get_default(model_name: str):
        return dict(
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

    def __call__(
            self, train_args: Dict = None,
            my_train_args: Dict[str, Union[int, str]] = None, train_dataset: datasets.Dataset = None
    ):
        args = self._get_default(self.model_name)
        train_args_ = TrainArgs.model_name2preset[self.model_name][self.model_size]
        if 'batch_size' in train_args_:
            assert 'per_device_train_batch_size' not in train_args_ and 'per_device_eval_batch_size' not in train_args_
            bsz = train_args_.pop('batch_size')
            train_args_['per_device_train_batch_size'] = train_args_['per_device_eval_batch_size'] = bsz
        args.update(train_args_)
        if self.model_name == 'xl':
            assert not args['fp16'] and not args['gradient_checkpointing']
        elif self.model_name == 'reformer':
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
    return TrainArgs(model_name, model_size)(train_args, my_train_args, train_dataset)


class ComputeMetrics:
    def __init__(self, tokenizer: MusicTokenizer, mode: str = 'vanilla'):
        self.acc = datasets.load_metric('accuracy')
        self.ikr = metrics.IkrMetric(tokenizer=tokenizer, mode=mode)
        self.mode = mode

    def __call__(self, eval_pred):
        """
        :param eval_pred: 2-tuple of (greedy prediction **ids**, labels)

        Will be the outputs on eval dataset, see `Trainer.compute_metrics`
        """
        if self.mode == 'vanilla':
            preds, labels, key_scores = eval_pred
        else:
            (preds, labels), key_scores = eval_pred, None
        preds = preds.argmax(axis=-1)
        d_metric = dict(ikr=self.ikr(preds=preds, labels=labels, key_scores=key_scores))

        preds, labels = preds[:, :-1], labels[:, 1:]  # shift for CLM
        labels, preds = labels.flatten(), preds.flatten()
        msk_non_pad = (labels != train_util.PT_LOSS_PAD)
        labels, preds = labels[msk_non_pad], preds[msk_non_pad]
        d_metric['ntp_acc'] = self.acc.compute(predictions=preds, references=labels)['accuracy']
        return d_metric


class VanillaMap:
    """
    Class instead of local function for pickling

    Map for vanilla training where keys need to be separately passed
    """
    n_key = len(key_ordinal2str)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        ret = self.tokenizer(samples['score'], padding='max_length', truncation=True)
        keys: List[Dict[str, Optional[float]]] = samples['keys']
        # convert to a tensor format to eventually pass down to `compute_loss` and `compute_metrics`
        # -1 for metric computation to ignore
        ret['key_scores'] = [[(d[key_ordinal2str[i]] or -1) for i in range(VanillaMap.n_key)] for d in keys]
        return ret


def get_all_setup(
        model_name: str, model_size: str,
        dataset_names: Union[str, List[str]], prec: int = 5, n_sample=None,
        model_config: Dict = None, train_args: Dict = None, my_train_args: Dict = None
) -> Tuple[torch.nn.Module, MusicTokenizer, Trainer]:
    # n_sample mainly for debugging
    tokenizer, model_, meta = get_model_n_tokenizer(model_name, model_size, prec=prec, model_config=model_config)
    my_train_args = my_train_args or dict()
    aug_key = my_train_args.pop('augment_key', False)
    if aug_key:
        # For now, just do non-deterministic sampling for eval set too, TODO?
        dset = KeySampleDataset.from_hf(dataset_names, tokenizer=tokenizer, get_dataset_kwargs=dict(n_sample=n_sample))
    else:
        dset = get_dataset(
            dataset_names=dataset_names, map_func=VanillaMap(tokenizer),
            remove_columns=['title', 'score', 'keys'], n_sample=n_sample  # i.e. keep the input ids only
        )
    tr, vl = dset['train'], dset['test']
    args, my_args, = get_train_and_my_train_args(model_name, model_size, train_args, my_train_args, tr)
    assert all(  # Ensure compatibility of dataset & tokenizer, see `music_export`
        get(json.loads(ds.info.description), 'extractor_meta.precision') == tokenizer.precision for ds in dset.values()
    )

    clm_acc_logging = True
    cm = ComputeMetrics(tokenizer=tokenizer, mode='aug-key' if aug_key else 'vanilla')
    # if key not augmented, need to pass key info to each sample for eval
    cls = train_util.MyTrainer if aug_key else train_util.MyEvalTrainer
    trainer_ = cls(
        model_meta=meta,
        clm_acc_logging=clm_acc_logging, my_args=my_args,
        train_metrics=dict(ikr=cm.ikr),  # TODO: calculate IRK when key not given?
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

    seed = sconfig('random-seed')

    def train_reformer(resume_from_checkpoint: str = None):
        # not set seed if reformer for LSH attention,
        # see https://huggingface.co/docs/transformers/model_doc/reformer#transformers.ReformerConfig.hash_seed
        md_nm = 'reformer'
        md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        # md_sz = 'small'
        # md_sz = 'base'
        ic(md_nm, md_sz)

        augment_key = False
        # if augment_key:
        dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, ' \
                  'meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
                  'meta={mode=melody, prec=5, th=1}, 2022-04-17_11-52-15'
        # else:
        #     dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, ' \
        #               'meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
        #     dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
        #               'meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52'
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
    # train_reformer()

    # checkpoint_path = os_join(PATH_BASE, DIR_PROJ, DIR_MDL, 'reformer', '2022-04-03_00-20-53', 'checkpoint-1856')
    # train(resume_from_checkpoint=checkpoint_path)

    def train_xl():
        md_nm = 'transf-xl'
        transformers.set_seed(seed)
        md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        n = 64
        # n = None
        max_length = 2048
        # max_length = None

        augment_key = False

        dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
        dnm_mst = 'musicnlp music extraction, dnm=MAESTRO, n=1276, ' \
                  'meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-28'
        dnms = [dnm_909, dnm_mst]
        my_train_args = dict(
            augment_key=augment_key,
            save_epochs=2
        )
        train_args = dict()
        if 'debug' not in md_sz:
            my_train_args.update(dict(tqdm=True, logging_strategy='epoch'))
            train_args = dict(per_device_train_batch_size=32)
        mdl, tokenizer, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, dataset_names=dnms, model_config=dict(max_length=max_length),
            n_sample=n, train_args=train_args, my_train_args=my_train_args
        )
        # ignore so that `None` don't get detached
        ignore_keys_for_eval = ['losses', 'mems', 'hidden_states', 'attentions']
        trainer.train(ignore_keys_for_eval=ignore_keys_for_eval)
    train_xl()
