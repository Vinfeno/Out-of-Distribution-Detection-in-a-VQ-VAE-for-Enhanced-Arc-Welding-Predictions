import argparse
import os
import logging as log
import torch
import matplotlib
from tqdm import tqdm
from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule
from dataloader.latentspace_dataloader import LatentPredDataModule
from dataloader.utils import (
    get_val_test_ids,
    get_val_test_experiments,
    get_experiment_ids,
    get_vs_val_test_ids,
    get_vd_val_test_ids,
    get_inv_vs_val_test_ids,
    get_inv_vd_val_test_ids,
)
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from model.mlp import MLP
from model.gru import GRU
from model.classification_model import ClassificationLightningModule


class Split_Trainer:
    def __init__(self) -> None:
        pass

    # def train_ex_split_models(
    #     self,
    #     epochs_list,
    #     num_embeddings,
    #     betas,
    #     current_ex_trained,
    #     total_trained,
    # ):
    #     for epochs in epochs_list:
    #         for embeddings in num_embeddings:
    #             for beta in betas:
    #                 os.system(
    #                     f'python3 train_recon_embed_ex_1_2_only.py --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="ex-split-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
    #                 )
    #                 current_ex_trained += 1
    #                 total_trained += 1
    #     return total_trained

    def train_model(
        self,
        hidden_dim=32,
        learning_rate=0.001,
        epochs=1,
        clipping_value=0.7,
        batch_size=512,
        dropout_p=0.1,
        num_embeddings=256,
        embedding_dim=32,
        n_resblocks=1,
        dataset="asimow",
        model_name="VQ-VAE-Patch",
        patch_size=25,
        batchnorm=True,
        checkpoint_name="default",
        beta=0.25,
        data_split=None,
        checkpoint_path="model_checkpoints/VQ-VAE-Patch/",
    ):
        if data_split is None:
            raise ValueError("Data split must be specified.")
        logger = CSVLogger("logs", name="vq-vae-transformer")
        input_dim = 2 if dataset == "asimow" else 1
        # load data
        if data_split == "ex":
            dataset_dict = get_val_test_experiments([1, 2])
            val_ids = dataset_dict["val_ids"]
            # test_ids = dataset_dict["test_ids"]
            test_ids = get_experiment_ids(3)
        elif data_split == "vs":
            val_ids = get_vs_val_test_ids()["val_ids"]
            test_ids = get_vs_val_test_ids()["test_ids"]
        elif data_split == "vd":
            val_ids = get_vd_val_test_ids()["val_ids"]
            test_ids = get_vd_val_test_ids()["test_ids"]
        elif data_split == "vs-inv":
            val_ids = get_inv_vs_val_test_ids()["val_ids"]
            test_ids = get_inv_vs_val_test_ids()["test_ids"]
        elif data_split == "vd-inv":
            val_ids = get_inv_vd_val_test_ids()["val_ids"]
            test_ids = get_inv_vd_val_test_ids()["test_ids"]

        val_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in val_ids]
        test_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in test_ids]
        data_module = ASIMoWDataModule(
            task="reconstruction",
            batch_size=batch_size,
            n_cycles=1,
            val_data_ids=val_ids,
            test_data_ids=test_ids,
        )
        if data_split == "ex":
            split_type = "ExperimentSplit"
        else:
            split_type = "ParameterSplit"

        input_dim = 2
        data_module.setup(stage="fit")
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
        model_checkpoint_name = f"{model_name}-{dataset}-{checkpoint_name}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path + split_type + "/",
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
            callbacks=[checkpoint_callback],
        )
        # trainer.test(model=model, datamodule=data_module)

    def train_multi_models(
        self,
        data_splits: list,
        epochs_list: list,
        num_embeddings: list,
        betas: list,
        save_path="model_checkpoints/VQ-VAE-Patch/",
    ):
        torch.set_float32_matmul_precision("medium")
        total_iterations = (
            len(data_splits) * len(epochs_list) * len(num_embeddings) * len(betas)
        )
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        for split in data_splits:
            for epochs in epochs_list:
                for beta in betas:
                    for embeddings in num_embeddings:
                        self.train_model(
                            epochs=epochs,
                            beta=beta,
                            data_split=split,
                            num_embeddings=embeddings,
                            checkpoint_name=f"{split}-split-epochs={epochs}-nEmb={embeddings}-beta={beta}",
                            checkpoint_path=save_path,
                        )
                        progress_bar.update(1)
        progress_bar.close()
