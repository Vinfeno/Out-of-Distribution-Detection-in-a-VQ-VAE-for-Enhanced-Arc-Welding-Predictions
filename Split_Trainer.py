import argparse
import os
import logging as log
import torch
import matplotlib
from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule
from dataloader.latentspace_dataloader import LatentPredDataModule
from dataloader.utils import (
    get_val_test_ids,
    get_val_test_experiments,
    get_experiment_ids,
)
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from model.mlp import MLP
from model.gru import GRU
from model.classification_model import ClassificationLightningModule


class Split_Trainer:
    def __init__(self) -> None:
        pass

    def train_ex_split_models(
        self,
        epochs_list,
        num_embeddings,
        betas,
        num_ex_models,
        current_ex_trained,
        total,
        total_trained,
    ):
        for epochs in epochs_list:
            for embeddings in num_embeddings:
                for beta in betas:
                    os.system(
                        f'python3 train_recon_embed_ex_1_2_only.py --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="ex-split-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
                    )
                    current_ex_trained += 1
                    total_trained += 1
        return total_trained

    def train_model(
        self,
        hidden_dim,
        learning_rate,
        epochs,
        clipping_value,
        batch_size,
        dropout_p,
        num_embeddings,
        embedding_dim,
        n_resblocks,
        dataset,
        model_name,
        decoder_type,
        patch_size,
        batchnorm,
        checkpoint_name,
        beta,
    ):
        logger = CSVLogger("logs", name="vq-vae-transformer")

        input_dim = 2 if dataset == "asimow" else 1

        # load data
        dataset_dict = get_val_test_experiments([1, 2])
        val_ids = dataset_dict["val_ids"]
        # test_ids = dataset_dict["test_ids"]
        test_ids = get_experiment_ids(3)
        logger.log_hyperparams(
            {
                "val_ids": str(val_ids),
                "test_ids": str(test_ids),
                "dataset-name": dataset,
                "model_name": model_name,
            }
        )
        log.info(f"Val ids: {val_ids}")
        log.info(f"Test ids: {test_ids}")

        if dataset == "asimow":
            val_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in val_ids]
            test_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in test_ids]
            data_module = ASIMoWDataModule(
                task="reconstruction",
                batch_size=batch_size,
                n_cycles=1,
                val_data_ids=val_ids,
                test_data_ids=test_ids,
            )
            input_dim = 2
        else:
            raise ValueError(f"Invalid dataset name: {dataset}")
        data_module.setup(stage="fit")
        train_loader_size = len(data_module.train_ds)
        log.info(f"Loaded Data - Train dataset size: {train_loader_size}")

        if model_name == "VQ-VAE":
            model = VectorQuantizedVAE(
                logger=logger,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                n_resblocks=n_resblocks,
                learning_rate=learning_rate,
                decoder_type=decoder_type,
                dropout_p=dropout_p,
            )
        elif model_name == "VQ-VAE-Patch":
            model = VQVAEPatch(
                logger=logger,
                hidden_dim=hidden_dim,
                input_dim=input_dim,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                n_resblocks=n_resblocks,
                learning_rate=learning_rate,
                dropout_p=dropout_p,
                patch_size=patch_size,
                beta=beta,
            )
        elif model_name == "Var_Batch_Size":
            model = VQVAEPatch(
                logger=logger,
                hidden_dim=hidden_dim,
                input_dim=input_dim,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                n_resblocks=n_resblocks,
                learning_rate=learning_rate,
                dropout_p=dropout_p,
                patch_size=patch_size,
            )
        elif model_name == "Var_Batch_No_Norm":
            model = VQVAEPatch(
                logger=logger,
                hidden_dim=hidden_dim,
                input_dim=input_dim,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                n_resblocks=n_resblocks,
                learning_rate=learning_rate,
                dropout_p=dropout_p,
                patch_size=patch_size,
            )
        else:
            raise ValueError("Invalid model name")

        model_checkpoint_name = f"{model_name}-{dataset}-{checkpoint_name}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"model_checkpoints/{model_name}/",
            monitor="val/loss",
            mode="min",
            filename=model_checkpoint_name,
        )
        early_stop_callback = EarlyStopping(
            monitor="val/loss", min_delta=0.0001, patience=10, verbose=False, mode="min"
        )

        trainer = Trainer(
            devices=1,
            num_nodes=1,
            max_epochs=epochs,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            gradient_clip_val=clipping_value,
        )

        trainer.fit(
            model=model,
            datamodule=data_module,
            batchnorm=batchnorm,
        )

        trainer = Trainer(
            devices=1,
            num_nodes=1,
            logger=logger,
            callbacks=[checkpoint_callback],
        )

        trainer.test(model=model, datamodule=data_module)

    def train_param_split_models(
        self,
        data_splits,
        epochs_list,
        num_embeddings,
        betas,
        num_param_models,
        current_param_trained,
        total,
        total_trained,
    ):
        for split in data_splits:
            for epochs in epochs_list:
                for beta in betas:
                    for embeddings in num_embeddings:
                        os.system(
                            f'python3 train_recon_embed_selected_runs.py --parameter="{split}" --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="{split}-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
                        )
                        current_param_trained += 1
                        total_trained += 1

        return total_trained

    def train(self) -> None:
        data_splits = hparams.data_splits
        epochs_list = hparams.epochs
        num_embeddings = hparams.num_embeddings
        betas = hparams.betas
        train = hparams.train
        eval = hparams.eval
        param_split = hparams.param_split
        ex_split = hparams.ex_split

        if (not train and not eval) or (not param_split and not ex_split):
            print("Nothing to do.")

        num_param_models = (
            len(data_splits) * len(epochs_list) * len(num_embeddings) * len(betas)
        )
        num_ex_models = len(epochs_list) * len(num_embeddings) * len(betas)
        if not param_split and ex_split:
            total = num_ex_models
        elif not ex_split and param_split:
            total = num_param_models
        else:
            total = num_param_models + num_ex_models

        total_trained = 0
        total_evaled = 0
        current_param_trained = 0
        current_param_evaled = 0
        current_ex_trained = 0
        current_ex_evaled = 0

        if train:
            if param_split:
                total_trained = train_param_split_models(
                    data_splits,
                    epochs_list,
                    num_embeddings,
                    betas,
                    num_param_models,
                    current_param_trained,
                    total,
                    total_trained,
                )
            if ex_split:
                total_trained = train_ex_split_models(
                    epochs_list,
                    num_embeddings,
                    betas,
                    num_ex_models,
                    current_ex_trained,
                    total,
                    total_trained,
                )
        if eval:
            if param_split:
                total_evaled = eval_param_split_models(
                    data_splits,
                    epochs_list,
                    num_embeddings,
                    betas,
                    num_param_models,
                    current_param_evaled,
                    total,
                    total_evaled,
                )
            if ex_split:
                total_evaled = eval_ex_split_models(
                    epochs_list,
                    num_embeddings,
                    betas,
                    num_ex_models,
                    current_ex_evaled,
                    total,
                    total_evaled,
                )
