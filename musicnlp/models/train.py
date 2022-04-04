"""
Proposed method: on compact melody & bass representation for autoregressive music generation
    Trying Transformer-XL, Reformer
"""
from copy import deepcopy

import torch
from transformers import TransfoXLConfig, ReformerConfig, ReformerModelWithLMHead
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer
from transformers.training_args import OptimizerNames
import datasets

from musicnlp.util import *
import musicnlp.util.train as train_util
import musicnlp.util.models as model_util
from musicnlp.preprocess import get_dataset
from musicnlp.models import MusicTokenizer, _models


def get_model_n_tokenizer(
        model_name: str, model_size: str, prec: int = 5, model_config: Dict = None
) -> Tuple[MusicTokenizer, model_util.MusicTransformerMixin, OrderedDict]:
    assert model_name in ['xl', 'reformer'], f'Unknown model_name: {model_name}'

    tokenizer_ = MusicTokenizer(prec=prec)  # needed for reformer config
    if not hasattr(get_model_n_tokenizer, 'd_config'):
        layer_pair = ['local', 'lsh']
        get_model_n_tokenizer.d_config = dict(
            xl={
                'debug': dict(d_model=8),
                'debug-large': dict(d_model=512),
                'small': dict(d_model=1024)
            },
            reformer={
                'debug': dict(max_position_embeddings=1024, axial_pos_shape=(32, 32)),
                # overall, given hidden size, keep
                #   feed_forward_size = 4 x hidden size
                #   attention_head_size = hidden_size
                'tiny': dict(
                    max_position_embeddings=1024, axial_pos_shape=(32, 32),
                    hidden_size=256, feed_forward_size=256*4, axial_pos_embds_dim=(64, 192),
                    # note attention head size in config is per head
                    num_attention_heads=8, attention_head_size=int(256/8),
                    # effectively 6 layers as default config; going even smaller produces an error
                    attn_layers=layer_pair*3
                ),
                'small': dict(
                    max_position_embeddings=2048, axial_pos_shape=(32, 64),
                    hidden_size=512, feed_forward_size=512*4, axial_pos_embds_dim=(128, 512-128),
                    num_attention_heads=8, attention_head_size=int(512/8),
                    attn_layers=layer_pair*3
                ),
                'base': dict(
                    max_position_embeddings=2048, axial_pos_shape=(32, 64),
                    hidden_size=768, feed_forward_size=768*4, axial_pos_embds_dim=(192, 768-192),
                    num_attention_heads=12, attention_head_size=int(768/12),
                    attn_layers=layer_pair*6,
                    num_hashes=2  # for better accuracy
                ),
                'large': dict(
                    max_position_embeddings=2048, axial_pos_shape=(32, 64),  # TODO: support token length 4096?
                    hidden_size=1024, feed_forward_size=1024*4, axial_pos_embds_dim=(256, 1024-256),
                    num_attention_heads=16, attention_head_size=int(1024/16),
                    attn_layers=layer_pair*12,
                    num_hashes=2
                )
            }
        )
        d_ref = get_model_n_tokenizer.d_config['reformer']
        for k in d_ref.keys():
            d_ref[k].update(dict(  # default config for all reformer config
                is_decoder=True,
                num_buckets=None,
                eos_token_id=tokenizer_.eos_token_id,
                pad_token_id=tokenizer_.pad_token_id,
                vocab_size=tokenizer_.vocab_size
            ))
    if not hasattr(get_model_n_tokenizer, 'd_nm2cls'):
        get_model_n_tokenizer.d_nm2cls = {
            'xl': (TransfoXLConfig, _models.MyTransfoXLLMHeadModel),
            'reformer': (ReformerConfig, ReformerModelWithLMHead)
        }
    cls_config, cls_model = get_model_n_tokenizer.d_nm2cls[model_name]
    config_ = cls_config()
    config_.update(get_model_n_tokenizer.d_config[model_name][model_size])
    if model_config is not None:
        config_.update(model_config)
    tokenizer_.model_max_length = max_length = model_util.config2model_size(config_)
    model_meta = OrderedDict([
        ('model name', cls_model.__qualname__),
        ('max length', max_length)
    ])
    if isinstance(config_, ReformerConfig):
        model_meta.update(dict(
            axial_pos_shape=config_.axial_pos_shape,
            n_layer=len(config_.attn_layers),
            hidden_size=config_.hidden_size, ff_size=config_.feed_forward_size,
            attention_shape=f'{config_.num_attention_heads}x{config_.attention_head_size}',
        ))
    else:
        raise NotImplementedError(f'xl')

    d_ref = get_model_n_tokenizer.d_config['reformer']
    for k in d_ref.keys():
        aps, mpe = d_ref[k]['axial_pos_shape'], d_ref[k]['max_position_embeddings']
        assert len(aps) == 2 and np.prod(aps) == mpe, \
            'the product of `axial_pos_shape` must be `max_position_embeddings`'
    # to set the correct model config for reformer, now take care of `max_length` for tokenizer
    return tokenizer_, cls_model(config=config_), model_meta  # Initialize all weights from scratch


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
    :param train_dataset: Training dataset, intended to get size for step per epoch calculation
    """
    if not hasattr(get_train_and_my_train_args, 'default_args'):
        get_train_and_my_train_args.default_args = dict(
            output_dir=os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, now(for_path=True)),
            do_train=True, do_eval=False,
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
            # gradient_checkpointing=torch.cuda.is_available()
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

    my_args: Dict[str, Union[int, str]] = dict()
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
    if 'logging_strategy' in my_args and my_args['logging_strategy'] == 'epoch':
        my_args['logging_steps'] = steps_per_epoch
    args = {k: v for k, v in args.items() if v is not None}
    return TrainingArguments(**args), my_args


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
    msk_non_pad = (labels != train_util.PT_LOSS_PAD)
    labels, predictions = labels[msk_non_pad], predictions[msk_non_pad]
    return compute_metrics.metric.compute(predictions=predictions, references=labels)


def get_all_setup(
        model_name: str, model_size: str, dataset_name: str, prec: int = 5, n_sample=None, dataset_seed=None,
        model_config: Dict = None, train_args: Dict = None, my_train_args: Dict = None
) -> Tuple[model_util.MusicTransformerMixin, MusicTokenizer, datasets.Dataset, Trainer]:
    tokenizer_, model_, meta = get_model_n_tokenizer(model_name, model_size, prec=prec, model_config=model_config)
    tr = get_dataset(
        dataset_name, map_func=lambda d: tokenizer_(d['text'], padding='max_length', truncation=True),
        remove_columns=['title', 'text'], n_sample=n_sample, random_seed=dataset_seed
    )
    args, my_args, = get_train_and_my_train_args(model_name, model_size, train_args, my_train_args, tr)
    # Ensure compatibility of dataset & tokenizer, see `music_export`
    assert json.loads(tr.info.description)['precision'] == tokenizer_.prec

    clm_acc_logging = isinstance(model_, ReformerModelWithLMHead)  # couldn't get logits for `TransfoXL`
    trainer_ = train_util.MyTrainer(
        model_meta=meta,
        clm_acc_logging=clm_acc_logging, my_args=my_args,
        model=model_, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False),
        train_dataset=tr, compute_metrics=compute_metrics
    )
    return model_, tokenizer_, tr, trainer_


if __name__ == '__main__':
    import transformers
    from icecream import ic

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'

    def check_model_size():
        md_nm = 'reformer'
        # md_sz = 'small'
        # md_sz = 'base'
        md_sz = 'large'
        mdl: torch.nn.Module = get_model_n_tokenizer(model_name=md_nm, model_size=md_sz)[1]
        ic(get_model_num_trainable_parameter(mdl))
    check_model_size()

    def train(resume_from_checkpoint: str = None):
        seed = config('random-seed')

        md_nm = 'reformer'
        md_sz = 'tiny'
        # md_sz = 'small'
        # md_sz = 'base'

        n = 8
        # n = None

        if md_nm != 'reformer':
            transformers.set_seed(seed)
        # not set seed if reformer for LSH attention,
        # see https://huggingface.co/docs/transformers/model_doc/reformer#transformers.ReformerConfig.hash_seed

        # train_args = dict(num_train_epochs=256)
        train_args = dict(
            per_device_train_batch_size=4,
            save_strategy='epoch',
            num_train_epochs=8
        )
        my_train_args = dict(
            logging_strategy='epoch',  # those are replicated from Trainer, for my own logging
            save_epochs=2
        )
        mdl, tokenizer, dset_tr, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, dataset_name=fnm, n_sample=n, dataset_seed=seed,
            train_args=train_args, my_train_args=my_train_args
        )
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint)
        else:
            trainer.train()
        trainer.save_model(os.path.join(trainer.args.output_dir, 'trained'))
    # train()

    # checkpoint_path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, 'reformer', '2022-04-03_00-20-53', 'checkpoint-1856')
    # train(resume_from_checkpoint=checkpoint_path)
