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
from musicnlp.vocab import MusicTokenizer
from musicnlp.preprocess import dataset, transform
from musicnlp.models import MyReformerConfig, MyReformerModelWithLMHead, MyTransfoXLConfig, MyTransfoXLLMHeadModel
from musicnlp.trainer import WordPieceMusicTokenizer, load_trained_tokenizer as load_wordpiece_tokenizer, metrics


def get_model_n_tokenizer(
        model_name: str, model_size: str, prec: int = 5, wordpiece_tokenize: bool = False, pitch_kind: str = None,
        model_config: Dict = None
) -> Tuple[MusicTokenizer, torch.nn.Module, OrderedDict]:
    ca.check_mismatch('Model Name', model_name, ['transf-xl', 'reformer'])
    if wordpiece_tokenize:
        fnm = wordpiece_tokenize if isinstance(wordpiece_tokenize, str) else None
        tokenizer: WordPieceMusicTokenizer = load_wordpiece_tokenizer(fnm=fnm, pitch_kind=pitch_kind)
        assert tokenizer.precision == prec
    else:
        tokenizer: MusicTokenizer = MusicTokenizer(precision=prec, pitch_kind=pitch_kind)
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
            ),
            'large': dict(
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
            ),
            'large': dict(
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

        my_args: Dict[str, Union[int, str]] = dict(
            logging_strategy='steps', tqdm=False, insert_key=False, proportional_mixing=False
        )
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
    def __init__(self, tokenizer: MusicTokenizer, mode: str = 'vanilla', clm_pred_shifted: bool = False):
        self.acc = evaluate.load('accuracy')
        self.ikr = metrics.IkrMetric(tokenizer=tokenizer, mode=mode, clm_pred_shifted=clm_pred_shifted)
        ca.check_mismatch('Training Mode', mode, ['vanilla', 'ins-key'])
        self.mode = mode

        self.clm_pred_shifted = clm_pred_shifted

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

        if not self.clm_pred_shifted:
            preds = preds[:, :-1]
        labels = labels[:, 1:]
        labels, preds = labels.flatten(), preds.flatten()
        msk_non_pad = (labels != train_util.PT_LOSS_PAD)
        labels, preds = labels[msk_non_pad], preds[msk_non_pad]
        d_metric['ntp_acc'] = self.acc.compute(predictions=preds, references=labels)['accuracy']
        return d_metric


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
    keys = [
        'random_crop', 'pitch_kind', 'insert_key', 'pitch_shift', 'channel_mixup',
        'wordpiece_tokenize', 'proportional_mixing'
    ]
    rand_crop, pch_kd, ins_key, pch_shift, mix_up, wp_tokenize, prop_mix = (my_train_args.get(k, False) for k in keys)
    logger.info(f'Loading model & tokenizer... ')
    tokenizer, model, meta = get_model_n_tokenizer(
        model_name, model_size, prec=prec, wordpiece_tokenize=wp_tokenize, pitch_kind=pch_kd,
        model_config=model_config
    )

    logger.info('Loading datasets... ')
    dset_args = dataset_args or dict()

    def load_once(dset_nms: Union[str, List[str]], split: str = None) -> Union[datasets.Dataset, datasets.DatasetDict]:
        get_dset_args = dict(splits=split) if split else dict()
        if rand_crop or pch_kd != 'step' or ins_key or pch_shift or mix_up:
            dset_args_ = dict(
                get_dataset_args=dset_args | get_dset_args, mode=my_train_args['mode'],
                random_crop=rand_crop,
                pitch_kind=pch_kd, insert_key=ins_key, channel_mixup=mix_up, pitch_shift=pch_shift
            ) | (dict(dataset_split=split) if split else dict())
            logger.info(f'Loading {pl.i("Augmented")} dataset w/ {pl.i(dict(dataset_names=dset_nms) | dset_args_)}... ')
            ret = dataset.AugmentedDataset.from_hf(dset_nms, tokenizer=tokenizer, **dset_args_)
        else:  # Dataset default pitch kind is `step`
            ret = dataset.get_dataset(
                dataset_names=dset_nms, map_func=transform.CombineKeys(tokenizer),
                remove_columns=['title', 'score', 'keys'], **(dset_args | get_dset_args)  # i.e. keep the input ids only
            )
        return ret[split] if split else ret
    if prop_mix:
        prop_mix = prop_mix if isinstance(prop_mix, int) else 2048
        dsets_tr = [load_once(dnm, split='train') for dnm in dataset_names]
        dsets_ts = [load_once(dnm, split='test') for dnm in dataset_names]
        dset = datasets.DatasetDict(dict(
            train=dataset.ProportionMixingDataset(dataset_list=dsets_tr, k=prop_mix),
            # test=load_once(dataset_names, split='test')
            test=dataset.ProportionMixingDataset(dataset_list=dsets_ts, k=prop_mix // 10)  # Smaller eval for speed
        ))
    else:
        dset = load_once(dataset_names)
    tr, vl = dset['train'], dset['test']
    args, my_args, = get_train_and_my_train_args(model_name, model_size, train_args, my_train_args, tr)
    assert all(  # Ensure compatibility of dataset & tokenizer, see `music_export`
        get(json.loads(ds.info.description), 'extractor_meta.precision') == tokenizer.precision for ds in dset.values()
    )

    cm = ComputeMetrics(tokenizer=tokenizer, mode='ins-key' if ins_key else 'vanilla')
    if ins_key:
        cls = train_util.MyTrainer
    else:  # if key not augmented, need to pass key info to each song for IKR logging
        # raise NotImplementedError(f'Update eval override after {pl.i("preprocess_logits_for_metrics")}')
        cls = train_util.MyEvalTrainer
    trainer_args_ = dict(
        model_meta=meta,
        monitor_ntp_acc=True, my_args=my_args,
        train_metrics=dict(ikr=cm.ikr),
        model=model, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=tr, eval_dataset=vl, compute_metrics=cm,
        preprocess_logits_for_metrics=max_out_logits
    )
    trainer_args_.update(trainer_args or dict())
    if 'transf-xl' in model_name and not trainer_args.get('disable_train_metrics', False):
        raise NotImplementedError()  # Logging additional train metrics makes GPU util low
    trainer_ = cls(**trainer_args_)
    return model, tokenizer, trainer_


if __name__ == '__main__':
    import transformers

    mic.output_width = 256

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
    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')
    # dnms = [pop]
    # dnms = [pop, mst]
    dnms = [pop, mst, lmd]

    def profile_transform_dataload():
        from tqdm.auto import tqdm, trange

        transformers.set_seed(seed)

        dsets = []
        tokenizer = load_wordpiece_tokenizer(pitch_kind='degree')
        tokenizer.model_max_length = 2048

        sp = 'train'
        for dnm in [pop, mst, lmd]:
            dsets.append(dataset.AugmentedDataset.from_hf(
                dnm, tokenizer=tokenizer, get_dataset_args=dict(shuffle_seed=seed, splits=sp),
                mode=md, random_crop=4, pitch_kind='degree', insert_key=True, channel_mixup=True, pitch_shift=True,
                dataset_split=sp
            )[sp])
        dset = dataset.ProportionMixingDataset(dataset_list=dsets, k=1280)
        # mic(dset.k, dset.dset_szs, dset.sz)

        for i in trange(len(dset)):  # Since `__iter__` not implemented
            _ = dset[i]
    # profile_runtime(profile_transform_dataload)

    def train_reformer(**kwargs):
        # not set seed if reformer for LSH attention,
        # see https://huggingface.co/docs/transformers/model_doc/reformer#transformers.ReformerConfig.hash_seed
        md_nm = 'reformer'
        md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        # md_sz = 'small'
        # md_sz = 'base'
        # md_sz = 'large'
        mic(md_nm, md_sz)

        # TODO: smaller seq-len for now, until it shows longer dependency
        model_config = None
        # model_config = dict(max_position_embeddings=1024, axial_pos_shape=(32, 32))

        insert_key = True
        pch_shift = True
        wordpiece_tokenize = False
        channel_mixup = 'full'
        # channel_mixup = False
        prop_mix = 1280
        mic(insert_key, pch_shift, wordpiece_tokenize, channel_mixup, prop_mix)

        n = 64
        # n = None
        # n_ep = 8
        # n_ep = 16
        # n_ep = 32
        # n_ep = 64
        n_ep = 128
        # n_ep = 512
        train_args = dict(save_strategy='epoch', num_train_epochs=n_ep)
        my_train_args = dict(
            tqdm=True, logging_strategy='no',
            mode=md, wordpiece_tokenize=wordpiece_tokenize, proportional_mixing=prop_mix,
            insert_key=insert_key, pitch_shift=pch_shift, channel_mixup=channel_mixup
        )
        trainer_args = dict(disable_train_metrics=True)

        if 'debug' not in md_sz:
            # if any('LMD' in d for d in dnms):  # Data includes LMD, a much larger dataset; but doesn't seem to help
            #     train_args['learning_rate'] = 3e-5
            if md_sz == 'base':
                bsz = 128 if on_great_lakes() else 64
            else:
                assert md_sz == 'large'
                bsz = 48
            train_args.update(dict(
                dataloader_num_workers=4,
                per_device_train_batch_size=bsz,
                per_device_eval_batch_size=bsz
            ))

        mdl, tokenizer, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, model_config=model_config,
            dataset_names=dnms, dataset_args=dict(n_sample=n, shuffle_seed=seed, pbar=True),
            train_args=train_args, my_train_args=my_train_args, trainer_args=trainer_args
        )
        trainer.train(**kwargs)
        trainer.save_model(os_join(trainer.args.output_dir, 'trained'))
    # train_reformer()

    def train_xl(**kwargs):  # TODO: support for disable NTP logging
        md_nm = 'transf-xl'
        # md_sz = 'debug'
        # md_sz = 'debug-large'
        # md_sz = 'tiny'
        # md_sz = 'base'
        md_sz = 'large'
        mic(md_nm, md_sz)

        debug = 'debug' in md_sz
        # debug = False

        # n = 8
        # n = 16
        # n = 8
        # n = 64
        # n = 128
        # n = 1024
        n = None
        # n_ep = 4
        # n_ep = 64
        n_ep = 128
        # n_ep = 256
        # n_ep = 512
        mic(n, n_ep)

        # model_config = dict(max_length=64)
        if debug:
            model_config = dict(
                max_length=1024 + 512,
                # cutoffs=[100]
            )
            pch_kd = 'midi'
            rand_crop, insert_key, pch_shift, wordpiece_tokenize, channel_mixup, prop_mix = (
                False, False, False, False, False, False
            )
        else:
            # model_config = None
            model_config = dict(
                max_length=1024,
                # mem_len=512,
            )  # TODO: try a smaller model for memory consumption
            # model_config = dict(max_length=1024 + 512)
            rand_crop = 4
            # pch_kd = 'midi'
            pch_kd = 'degree'
            insert_key = True
            # pch_shift = False
            pch_shift = True
            if pch_shift:
                assert insert_key and pch_kd == 'degree'
            else:
                assert pch_kd != 'degree'
            channel_mixup = 'full'
            # channel_mixup = False
            wordpiece_tokenize = False
            # wordpiece_tokenize = True
            # wordpiece_tokenize = '22-11-26_WordPiece-Tokenizer_{dnm=all}_{vsz=262144, n=178825, pch=d, aug-key=T}'
            if not wordpiece_tokenize:
                model_config['cutoffs'] = []
            # if pch_kd == 'midi':
            #     wordpiece_tokenize = ''
            # wordpiece_tokenize = '22-11-08_WordPiece-Tokenizer_{dnm=POP&MST}_{vsz=32768, n=2185, pch=d, aug-key=T}'
            # prop_mix = False
            prop_mix = 1280
        mic(rand_crop, pch_kd, insert_key, pch_shift, channel_mixup, wordpiece_tokenize, prop_mix)

        # needed so that best model is loaded in the end
        train_args = dict(save_strategy='epoch', num_train_epochs=n_ep)
        my_train_args = dict(
            tqdm=True, logging_strategy='no',
            mode=md,
            random_crop=rand_crop,
            pitch_kind=pch_kd, insert_key=insert_key, pitch_shift=pch_shift, channel_mixup=channel_mixup,
            wordpiece_tokenize=wordpiece_tokenize, proportional_mixing=prop_mix,
        )
        trainer_args = dict(disable_train_metrics=True)

        if debug:
            train_args.update(dict(
                learning_rate=1e-3,
                weight_decay=0,
                warmup_ratio=0,
                lr_scheduler_type=SchedulerType.CONSTANT,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
            ))
        else:
            # bsz = 24
            # bsz = 21
            bsz = 12
            train_args.update(dict(
                # learning_rate=1e-4,
                dataloader_num_workers=4,
                per_device_train_batch_size=bsz,
                per_device_eval_batch_size=bsz,
            ))
        mdl, tokenizer, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, model_config=model_config,
            dataset_names=dnms, dataset_args=dict(n_sample=n, shuffle_seed=seed),
            train_args=train_args, my_train_args=my_train_args, trainer_args=trainer_args
        )

        transformers.set_seed(seed)
        # ignore so that `None` don't get detached
        ignore_keys_for_eval = ['losses', 'mems', 'hidden_states', 'attentions']
        train_call_args = dict(ignore_keys_for_eval=ignore_keys_for_eval)
        trainer.train(**(train_call_args | kwargs))
        trainer.save_model(os_join(trainer.args.output_dir, 'trained'))
        # tokenizer.save_pretrained(os_join(trainer.args.output_dir, 'tokenizer'))  # TODO
    train_xl()
