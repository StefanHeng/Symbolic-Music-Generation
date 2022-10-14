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
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling, Trainer
from transformers.training_args import OptimizerNames
import datasets
import evaluate

from stefutil import *
from musicnlp.util import *
import musicnlp.util.train as train_util
from musicnlp.vocab import MusicTokenizer, key_ordinal2str
from musicnlp.preprocess import DATASET_NAME2MODE2FILENAME, get_dataset, AugmentedDataset
from musicnlp.models import MyReformerConfig, MyReformerModelWithLMHead, MyTransfoXLConfig, MyTransfoXLLMHeadModel
from musicnlp.trainer import WordPieceMusicTokenizer, load_trained_tokenizer as load_wordpiece_tokenizer, metrics


def get_model_n_tokenizer(
        model_name: str, model_size: str, prec: int = 5, wordpiece_tokenize: bool = False, model_config: Dict = None
) -> Tuple[MusicTokenizer, torch.nn.Module, OrderedDict]:
    ca.check_mismatch('Model Name', model_name, ['transf-xl', 'reformer'])
    if wordpiece_tokenize:
        tokenizer: WordPieceMusicTokenizer = load_wordpiece_tokenizer()
        assert tokenizer.precision == prec
    else:
        tokenizer: MusicTokenizer = MusicTokenizer(precision=prec)
    if not hasattr(get_model_n_tokenizer, 'd_nm2cls'):
        get_model_n_tokenizer.d_nm2cls = {
            'transf-xl': (MyTransfoXLConfig, MyTransfoXLLMHeadModel),
            'reformer': (MyReformerConfig, MyReformerModelWithLMHead)
        }
    cls_config, cls_model = get_model_n_tokenizer.d_nm2cls[model_name]
    config = cls_config(model_size=model_size, tokenizer=tokenizer, **(model_config or dict()))
    # to set the correct model config for reformer, now take care of `max_length` for tokenizer
    tokenizer.model_max_length = max_length = config.max_length_
    model_meta = OrderedDict({'model name': cls_model.cls_name, 'max length': max_length})
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
                num_train_epochs=64,
                warmup_ratio=0.1
            ),
            'small': dict(
                batch_size=32,
                learning_rate=3e-4,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                num_train_epochs=64,
                warmup_ratio=0.1
            ),
            'base': dict(
                batch_size=32,
                learning_rate=3e-4,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                num_train_epochs=64,
                warmup_ratio=0.1
            )
        },
        'reformer': {
            'debug': dict(
                batch_size=8,
                learning_rate=1e-3,
                weight_decay=0,
                lr_scheduler_type=SchedulerType.CONSTANT,
                num_train_epochs=32,
            ),
            'debug-large': dict(
                batch_size=8,
                learning_rate=1e-3,
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
                num_train_epochs=64,
                warmup_ratio=0.1
            ),
            'base': dict(
                batch_size=32,
                learning_rate=3e-4,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                num_train_epochs=64,
                warmup_ratio=0.1
            )
        }
    }

    def __init__(self, model_name: str, model_size: str):
        self.model_name, self.model_size = model_name, model_size

    @staticmethod
    def _get_default(model_name: str):
        return dict(
            output_dir=os_join(get_base_path(), u.model_dir, f'{now(for_path=True)}_{model_name}'),
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
            gradient_checkpointing=False,  # not supported in both Transformer XL and Reformer not supported
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
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
        assert not args['gradient_checkpointing']
        if train_args is not None:
            args.update(train_args)

        my_args: Dict[str, Union[int, str]] = dict(logging_strategy='steps', tqdm=False, augment_key=False)  # default
        my_args.update(my_train_args or dict())
        bsz = args['per_device_train_batch_size'] * args.get('gradient_accumulation_steps', 1)
        my_args['steps_per_epoch'] = steps_per_epoch = math.ceil(len(train_dataset) / bsz)
        save_epochs = my_args.get('save_epochs', None)
        if save_epochs:  # this is not supported by Trainer
            assert 'save_strategy' in args and args['save_strategy'] == 'epoch', \
                f'Supporting {pl.i("save per epoch")} error: Save strategy to Trainer should be set to {pl.i("epoch")}'
            if save_epochs > 1:
                args['save_strategy'] = 'steps'
                # TODO: DDP not supported
                args['save_steps'] = save_epochs * steps_per_epoch
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


def max_out_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Sorting logits of size (batch_size, seq_len, vocab_size) across entire eval set in RAM is too much & unnecessary
    """
    return logits.argmax(dim=-1)


class ComputeMetrics:
    def __init__(self, tokenizer: MusicTokenizer, mode: str = 'vanilla'):
        self.acc = evaluate.load('accuracy')
        self.ikr = metrics.IkrMetric(tokenizer=tokenizer, mode=mode)
        ca.check_mismatch('Training Mode', mode, ['vanilla', 'key-aug'])
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
        # preds = preds.argmax(axis=-1)  # No longer needed, see `preprocess_logits_for_metrics`
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
        model_name: str = None, model_size: str = None, model_config: Dict = None,
        dataset_names: Union[str, List[str]] = None, prec: int = 5, dataset_args: Dict = None,
        train_args: Dict = None, my_train_args: Dict = None, trainer_args: Dict = None
) -> Tuple[torch.nn.Module, MusicTokenizer, Trainer]:
    logger = get_logger('Get Setup')
    d_log = dict(
        model_name=model_name, model_size=model_size, model_config=model_config,
        dataset_names=dataset_names, prec=prec, dataset_args=dataset_args,
        train_args=train_args, my_train_args=my_train_args
    )
    logger.info(f'Initializing training with {pl.fmt(d_log)}... ')
    my_train_args = my_train_args or dict()
    wordpiece_tokenize, aug_key, mix_up = (
        my_train_args.get(k, False) for k in ['wordpiece_tokenize', 'augment_key', 'channel_mixup']
    )
    logger.info(f'Loading model & tokenizer... ')
    tokenizer, model, meta = get_model_n_tokenizer(
        model_name, model_size, prec=prec, wordpiece_tokenize=wordpiece_tokenize, model_config=model_config
    )

    logger.info('Loading datasets... ')
    dset_args = dataset_args or dict()
    if aug_key or mix_up:
        dset_args_ = dict(
            get_dataset_kwargs=dset_args, augment_key=aug_key, channel_mixup=mix_up, mode=my_train_args['mode']
        )
        logger.info(f'Loading {pl.i("Augmented")} dataset w/ {pl.i(dset_args_)}... ')
        dset = AugmentedDataset.from_hf(dataset_names, tokenizer=tokenizer, **dset_args_)
    else:
        dset = get_dataset(
            dataset_names=dataset_names, map_func=VanillaMap(tokenizer),
            remove_columns=['title', 'score', 'keys'], **dset_args  # i.e. keep the input ids only
        )
    tr, vl = dset['train'], dset['test']
    args, my_args, = get_train_and_my_train_args(model_name, model_size, train_args, my_train_args, tr)
    assert all(  # Ensure compatibility of dataset & tokenizer, see `music_export`
        get(json.loads(ds.info.description), 'extractor_meta.precision') == tokenizer.precision for ds in dset.values()
    )

    cm = ComputeMetrics(tokenizer=tokenizer, mode='key-aug' if aug_key else 'vanilla')
    # if key not augmented, need to pass key info to each song for IKR logging
    if aug_key:
        cls = train_util.MyTrainer
    else:
        raise NotImplementedError(f'Update eval override after {pl.i("preprocess_logits_for_metrics")}')
        # cls = train_util.MyEvalTrainer
    trainer_args_ = dict(
        model_meta=meta,
        monitor_ntp_acc=True, my_args=my_args,
        train_metrics=dict(ikr=cm.ikr),
        model=model, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=tr, eval_dataset=vl, compute_metrics=cm,
        preprocess_logits_for_metrics=max_out_logits
    )
    trainer_args_.update(trainer_args or dict())
    trainer_ = cls(**trainer_args_)
    return model, tokenizer, trainer_


if __name__ == '__main__':
    import transformers

    def check_model_size():
        md_nm = 'reformer'
        # md_sz = 'small'
        # md_sz = 'base'
        md_sz = 'large'
        mdl: torch.nn.Module = get_model_n_tokenizer(model_name=md_nm, model_size=md_sz)[1]
        mic(get_model_num_trainable_parameter(mdl))
    # check_model_size()

    seed = sconfig('random-seed')

    # md = 'melody'
    md = 'full'
    # dnms = ['LMD']
    # dnms = ['POP909', 'MAESTRO']
    dnms = ['POP909', 'MAESTRO', 'LMD']
    dnms = [get(DATASET_NAME2MODE2FILENAME, f'{dnm}.{md}') for dnm in dnms]

    def train_reformer(resume: str = None):
        # not set seed if reformer for LSH attention,
        # see https://huggingface.co/docs/transformers/model_doc/reformer#transformers.ReformerConfig.hash_seed
        md_nm = 'reformer'
        # md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        # md_sz = 'small'
        md_sz = 'base'
        mic(md_nm, md_sz)

        # TODO: smaller seq-len for now, until it shows longer dependency
        model_config = dict(max_position_embeddings=1024, axial_pos_shape=(32, 32))

        # augment_key = False
        augment_key = True
        # wordpiece_tokenize = False
        wordpiece_tokenize = True
        # channel_mixup = False
        channel_mixup = True

        # _debug_eval = True
        _debug_eval = False
        mic(augment_key, wordpiece_tokenize, channel_mixup, _debug_eval)

        # n_ep = 8
        n_ep = 16
        # n_ep = 32
        # n_ep = 256
        train_args = dict(save_strategy='epoch', num_train_epochs=n_ep)
        if not _debug_eval and channel_mixup:
            train_args['dataloader_num_workers'] = 4

        my_train_args = dict(
            tqdm=True, logging_strategy='epoch',
            augment_key=augment_key,
            wordpiece_tokenize=wordpiece_tokenize,
            channel_mixup=channel_mixup,
            mode=md
        )

        if 'debug' in md_sz or md_sz == 'tiny':
            train_args.update(dict(
                per_device_train_batch_size=4,
                num_train_epochs=32
            ))
            my_train_args['save_epochs'] = 16
        else:
            if any('LMD' in d for d in dnms):  # Data includes LMD, a much larger dataset
                train_args['learning_rate'] = 1e-5
            bsz = 128 if on_great_lakes() else 64
            train_args.update(dict(
                fp16=torch.cuda.is_available(),
                per_device_train_batch_size=bsz,
                per_device_eval_batch_size=bsz
            ))
            my_train_args.update(dict(
                logging_strategy='no',
                # save_epochs=8
            ))

        # n = 64
        n = None

        mdl, tokenizer, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, model_config=model_config,
            dataset_names=dnms, dataset_args=dict(n_sample=n, shuffle_seed=seed),
            train_args=train_args, my_train_args=my_train_args, trainer_args=dict(
                disable_train_metrics=True
            )
        )
        if _debug_eval:
            # trainer.train_dataset: datasets.Dataset
            trainer.train_dataset: AugmentedDataset
            trainer.train_dataset.dset = trainer.train_dataset.dset.select(range(0, 256))
            mic(len(trainer.train_dataset))

        if resume:
            trainer.train(resume)
        else:
            trainer.train()
        trainer.save_model(os_join(trainer.args.output_dir, 'trained'))
    train_reformer()

    # checkpoint_path = os_join(PATH_BASE, DIR_PROJ, DIR_MDL, 'reformer', '2022-04-03_00-20-53', 'checkpoint-1856')
    # train(resume_from_checkpoint=checkpoint_path)

    def train_xl(resume: str = None):  # TODO: support for disable NTP logging
        transformers.set_seed(seed)

        md_nm = 'transf-xl'

        md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        # md_sz = 'base'

        # n = 8
        # n = 16
        n = 128
        # n = 1024
        # n = None

        gas = 1
        # gas = 4
        n_ep = 4

        # mem_len = 256
        mem_len = None
        # max_length = 512
        # max_length = 1024
        max_length = None  # 2048 for non-debugging configs
        model_config = dict(max_length=max_length)

        augment_key = False
        # wordpiece_tokenize = False
        wordpiece_tokenize = True

        train_args = dict(gradient_accumulation_steps=gas)
        my_train_args = dict(
            augment_key=augment_key,
            wordpiece_tokenize=wordpiece_tokenize,
            save_epochs=1
        )

        # with_tqdm = False
        with_tqdm = True
        if with_tqdm:
            my_train_args.update(dict(tqdm=True, logging_strategy='epoch'))

        if 'debug' in md_sz:
            train_args['num_train_epochs'] = n_ep
        else:
            if mem_len:
                model_config['mem_len'] = mem_len
            train_args.update(dict(
                per_device_train_batch_size=20,
                per_device_eval_batch_size=20,
                num_train_epochs=n_ep
            ))
        mdl, tokenizer, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, model_config=model_config,
            dataset_names=dnms, dataset_args=dict(n_sample=n, shuffle_seed=seed),
            train_args=train_args, my_train_args=my_train_args, trainer_args=dict(
                disable_train_metrics=True
            )
        )
        sanity_check_eval = True
        if sanity_check_eval:
            trainer.train_dataset: datasets.Dataset
            trainer.train_dataset = trainer.train_dataset.select(range(8))
            mic(len(trainer.train_dataset))
        # ignore so that `None` don't get detached
        ignore_keys_for_eval = ['losses', 'mems', 'hidden_states', 'attentions']
        train_call_args = dict(ignore_keys_for_eval=ignore_keys_for_eval)
        if resume:
            train_call_args['resume_from_checkpoint'] = resume
        trainer.train(**train_call_args)
        trainer.save_model(os_join(trainer.args.output_dir, 'trained'))
    # train_xl()
    # with a large vocab size for WordPiece tokenizer, e.g. 16K, nested concat is the bottleneck in eval
    # profile_runtime(train_xl)
    # resume = os_join(u.model_path, 'checkpoint-17526')
    # train_xl(resume)
