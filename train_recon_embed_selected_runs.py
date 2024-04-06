import argparse
import os
import logging as log
import torch
import matplotlib
from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule
from dataloader.latentspace_dataloader import LatentPredDataModule
from dataloader.utils import (
    get_vs_val_test_ids,
    get_vd_val_test_ids,
    get_inv_vs_val_test_ids,
    get_inv_vd_val_test_ids,
)
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from model.mlp import MLP
from model.gru import GRU

def main(hparams):
    # read hyperparameters
    hidden_dim = hparams.hidden_dim
    learning_rate = hparams.learning_rate
    epochs = hparams.epochs
    clipping_value = hparams.clipping_value
    batch_size = hparams.batch_size
    dropout_p = hparams.dropout_p
    num_embeddings = hparams.num_embeddings
    embedding_dim = hparams.embedding_dim
    n_resblocks = hparams.n_resblocks
    dataset = hparams.dataset
    model_name = hparams.model_name
    decoder_type = hparams.decoder_type
    patch_size = hparams.patch_size
    checkpoint_name = hparams.checkpoint_name
    parameter = hparams.parameter

    use_wandb = hparams.use_wandb
    wandb_entity = hparams.wandb_entity
    wandb_project = hparams.wandb_project
    beta = hparams.beta

    if use_wandb:
        assert wandb_entity is not None, "Wandb entity must be set"
        assert wandb_project is not None, "Wandb project must be set"
        logger = WandbLogger(log_model=True, project=wandb_project, entity=wandb_entity)
    else:
        logger = CSVLogger("logs", name="vq-vae-transformer")

    input_dim = 2 if dataset == "asimow" else 1

    # load data
    if parameter == "vs":
        val_ids = get_vs_val_test_ids()["val_ids"]
        test_ids = get_vs_val_test_ids()["test_ids"]
    elif parameter == "vd":
        val_ids = get_vd_val_test_ids()["val_ids"]
        test_ids = get_vd_val_test_ids()["test_ids"]
    elif parameter == "vs-inv":
        val_ids = get_inv_vs_val_test_ids()["val_ids"]
        test_ids = get_inv_vs_val_test_ids()["test_ids"]
    elif parameter == "vd-inv":
        val_ids = get_inv_vd_val_test_ids()["val_ids"]
        test_ids = get_inv_vd_val_test_ids()["test_ids"]
    else:
        raise ValueError("Select vs[-inv] or vd[-inv].")

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
    )

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train", default=1
    )
    parser.add_argument("--dataset", type=str, help="Dataset", default="asimow")
    parser.add_argument(
        "--num-embeddings", type=int, help="Number of embeddings", default=256
    )
    parser.add_argument(
        "--embedding-dim", type=int, help="Dimension of one embedding", default=32
    )
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension", default=32)
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate", default=0.001
    )
    parser.add_argument(
        "--clipping-value", type=float, help="Gradient Clipping", default=0.7
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", default=512)
    parser.add_argument(
        "--beta", type=float, help="Commitment loss weight", default=0.25
    )
    parser.add_argument(
        "--n-resblocks", type=int, help="Number of Residual Blocks", default=1
    )
    parser.add_argument(
        "--patch-size", type=int, help="Patch size of the VQ-VAE Encoder", default=25
    )
    parser.add_argument(
        "--dropout-p", type=float, help="Dropout probability", default=0.1
    )
    parser.add_argument(
        "--model-name", type=str, help="Model name", default="VQ-VAE-Patch"
    )
    parser.add_argument(
        "--decoder-type", type=str, help="VQ-VAE Decoder Type", default="Conv"
    )

    parser.add_argument(
        "--checkpoint-name", type=str, help="Checkpoint name", default="default"
    )
    parser.add_argument(
        "--use-wandb",
        help="Use Weights and Bias (https://wandb.ai/) for Logging",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--wandb-entity", type=str, help="Weights and Bias entity")
    parser.add_argument("--wandb-project", type=str, help="Weights and Bias project")
    parser.add_argument(
        "--parameter", type=str, help="Parameter to split on ['vs', 'vs-inv', 'vd', 'vd-inv']"
    )

    args = parser.parse_args()

    matplotlib.use("agg")

    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision("medium")
    main(args)
