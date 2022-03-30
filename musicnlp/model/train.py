"""
Proposed method: on compact melody & bass representation for autoregressive music generation
    Trying Transformer-XL, Reformer
"""

from transformers import TransfoXLConfig, ReformerConfig, ReformerModelWithLMHead
from transformers import TrainingArguments, SchedulerType, DataCollatorForLanguageModeling
from transformers import Trainer
from transformers.training_args import OptimizerNames
import datasets

from musicnlp.util import *
import musicnlp.util.train as train_util
import musicnlp.util.model as model_util
from musicnlp.preprocess import get_dataset
from musicnlp.model import MusicTokenizer, models


def get_model_n_tokenizer(
        model_name: str, model_size: str, prec: int = 5, model_config: Dict = None
) -> Tuple[MusicTokenizer, model_util.MusicTransformerMixin]:
    assert model_name in ['xl', 'reformer'], f'Unknown model_name: {model_name}'

    tokenizer_ = MusicTokenizer(prec=prec)  # needed for reformer config
    if not hasattr(get_model_n_tokenizer, 'd_config'):
        get_model_n_tokenizer.d_config = dict(
            xl={
                'debug': dict(d_model=8),
                'debug-large': dict(d_model=512),
                'small': dict(d_model=1024)
            },
            reformer={
                'debug': dict(max_position_embeddings=1024, axial_pos_shape=(32, 32)),
                'base': dict(max_position_embeddings=4096, axial_pos_shape=(64, 64))
            }
        )
        d_ref = get_model_n_tokenizer.d_config['reformer']
        for k in d_ref.keys():
            aps, mpe = d_ref[k]['axial_pos_shape'], d_ref[k]['max_position_embeddings']
            assert len(aps) == 2 and math.prod(aps) == mpe, \
                'the product of `axial_pos_shape` must be `max_position_embeddings`'
            d_ref[k] |= dict(  # default config for all reformer config
                is_decoder=True,
                num_buckets=None,
                eos_token_id=tokenizer_.eos_token_id,
                pad_token_id=tokenizer_.pad_token_id,
                vocab_size=tokenizer_.vocab_size
            )
    if not hasattr(get_model_n_tokenizer, 'd_nm2cls'):
        get_model_n_tokenizer.d_nm2cls = {
            'xl': (TransfoXLConfig, models.MyTransfoXLLMHeadModel),
            'reformer': (ReformerConfig, ReformerModelWithLMHead)
        }
    cls_config, cls_model = get_model_n_tokenizer.d_nm2cls[model_name]
    config_ = cls_config()
    config_.update(get_model_n_tokenizer.d_config[model_name][model_size])
    if model_config is not None:
        config_.update(model_config)

    d_ref = get_model_n_tokenizer.d_config['reformer']
    for k in d_ref.keys():
        aps, mpe = d_ref[k]['axial_pos_shape'], d_ref[k]['max_position_embeddings']
        assert len(aps) == 2 and math.prod(aps) == mpe, \
            'the product of `axial_pos_shape` must be `max_position_embeddings`'
    # to set the correct model config for reformer, now take care of `max_length` for tokenizer
    tokenizer_.model_max_length = model_util.config2model_size(config_)
    return tokenizer_, cls_model(config_)  # Initialize all weights from scratch


def get_train_args(model_name: str, model_size: str, train_args: Dict = None) -> TrainingArguments:
    train_args_ = dict(
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
                batch_size=4,
                learning_rate=3e-4,
                weight_decay=0,
                lr_scheduler_type=SchedulerType.CONSTANT,
                num_train_epochs=32,
            ),
            'base': dict(
                batch_size=128,
                learning_rate=3e-5,
                weight_decay=1e-2,
                lr_scheduler_type=SchedulerType.COSINE,
                warmup_ratio=0.1
            )
        }
    )
    bsz, lr, decay, sch, n_ep, gas = (train_args_[model_name][model_size].get(k, None) for k in (
        'batch_size', 'learning_rate', 'weight_decay',
        'lr_scheduler_type', 'num_train_epochs', 'gradient_accumulation_steps'
    ))
    args = dict(
        output_dir=os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, now(for_path=True)),
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
        model_config: Dict = None, train_args: Dict = None
) -> Tuple[model_util.MusicTransformerMixin, MusicTokenizer, datasets.Dataset, Trainer]:
    tokenizer_, model_ = get_model_n_tokenizer(model_name, model_size, prec=prec, model_config=model_config)
    args = get_train_args(model_name, model_size, train_args)
    tr = get_dataset(
        dataset_name, map_func=lambda d: tokenizer_(d['text'], padding='max_length', truncation=True),
        remove_columns=['title', 'text'], n_sample=n_sample, random_seed=dataset_seed
    )
    # Ensure compatibility of dataset & tokenizer, see `music_export`
    assert json.loads(tr.info.description)['precision'] == tokenizer_.prec

    trainer_ = train_util.MyTrainer(
        clm_acc_logging=False,  # TODO: get logits for transformer-xl?
        model=model_, args=args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False),
        train_dataset=tr, compute_metrics=compute_metrics
    )
    return model_, tokenizer_, tr, trainer_


if __name__ == '__main__':
    import transformers
    from icecream import ic

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'

    def train():
        seed = config('random-seed')

        md_nm = 'reformer'
        md_sz = 'debug'

        # n = 4
        n = None

        if md_nm != 'reformer':
            transformers.set_seed(seed)
        # not set seed if reformer for LSH attention,
        # see https://huggingface.co/docs/transformers/model_doc/reformer#transformers.ReformerConfig.hash_seed
        mdl, tokenizer, dset_tr, trainer = get_all_setup(
            model_name=md_nm, model_size=md_sz, dataset_name=fnm, n_sample=n, dataset_seed=seed
        )
        trainer.train()
        trainer.save_model(os.path.join(trainer.args.output_dir, 'final-trained'))
    train()
