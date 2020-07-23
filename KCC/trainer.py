from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger, model_checkpoint

from argparse import Namespace

from .kcc_lightning_model import KcBERT_Classifier

import os, sys
import torch


def train(
    file_path,
    train_ratio=0.8,
    optimizer="AdamW",
    intent_optimizer_lr=1e-4,
    entity_optimizer_lr=2e-4,
    intent_loss_weight=1.0,
    intent_center_loss_weight=0.1,
    entity_loss_weight=1.0,
    epochs=20,
    batch_size=None,
    gpu_num=0,
    distributed_backend=None,
    checkpoint_prefix='kcc_model_'
):
    early_stopping = EarlyStopping('val_loss')
    lr_logger = LearningRateLogger()
    checkpoint_callback = model_checkpoint.ModelCheckpoint(prefix=checkpoint_prefix)

    prepare_data_per_node=True
    if gpus != 0: prepare_data_per_node=False

    if batch_size is None:
        trainer = Trainer(
            auto_scale_batch_size="power",
            max_epochs=epochs,
            gpus=gpu_num,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stopping,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback,
            prepare_data_node=prepare_data_node
        )
    else:
        trainer = Trainer(
            max_epochs=epochs,
            gpus=gpu_num,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stopping,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback,
            prepare_data_node=prepare_data_node
        )

    model_args = {}
    model_args["epochs"] = epochs
    model_args["batch_size"] = batch_size
    model_args["nlu_data"] = open(file_path, encoding="utf-8").readlines()
    model_args["train_ratio"] = train_ratio
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr
    model_args["entity_optimizer_lr"] = entity_optimizer_lr

    hparams = Namespace(**model_args)
    model = KcBERT_Classifier(hparams)
    trainer.fit(model)
