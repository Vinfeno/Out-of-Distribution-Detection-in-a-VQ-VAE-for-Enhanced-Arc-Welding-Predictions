import argparse
import os
import logging as log
import torch
import wandb
import matplotlib
from data_loader.asimow_dataloader import DataSplitId, ASIMoWDataModule
from data_loader.latentspace_dataloader import LatentPredDataModule
from model.vq_vae_multitask import VectorQuantizedMultiTask
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from select_val_test_set import get_val_test_ids, get_out_of_dist_dataset
from model.mlp import MLP
from model.gru import GRU


def print_training_input_shape(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    for i in range(len(batch)):
        log.info(f"Input {i} shape: {batch[i].shape}")
    

def classify_latent_space(latent_model: VectorQuantizedMultiTask | VectorQuantizedVAE | VQVAEPatch, wandb_logger: WandbLogger, val_ids: list[DataSplitId], 
                          test_ids: list[DataSplitId], n_cycles: int, model_name: str, dataset: str, dataset_id: int,
                          classification_model: str, learning_rate: float, clipping_value: float):
    
        data_module = LatentPredDataModule(latent_space_model=latent_model, model_name=f"{model_name}", val_data_ids=val_ids, test_data_ids=test_ids,
                                       n_cycles=n_cycles, task='classification', batch_size=128, dataset_id=dataset_id, 
                                       wandb_model_name=f"{model_name}-{dataset}")	
        print_training_input_shape(data_module)

        seq_len = n_cycles
        input_dim = int(latent_model.embedding_dim * latent_model.enc_out_len)
        
        if classification_model == "MLP":
            Model = MLP
        elif classification_model == "GRU":
            Model = GRU
        else:
            raise ValueError(f"Invalid classification model name: {classification_model}")


        model = Model(input_size=seq_len, in_dim=input_dim, hidden_sizes=128, dropout_p=0.1,
                      n_hidden_layers=4, output_size=2, learning_rate=learning_rate, model_id=str(dataset_id))
        
        
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints", monitor=f"{dataset_id}/val/f1_score", mode="max", filename=f"{model_name}-{dataset}-best")
        early_stop_callback = EarlyStopping(
            monitor=f"{dataset_id}/val/f1_score", min_delta=0.0001, patience=10, verbose=False, mode="max")

        trainer = Trainer(
            max_epochs=15,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            devices=1,
            num_nodes=1,
            gradient_clip_val=clipping_value,
            check_val_every_n_epoch=1
        )

        trainer.fit(
            model=model,
            datamodule=data_module,
        )

        best_score = model.hyper_search_value
        print(f"best score: {best_score}")
        print("------ Testing ------")

        trainer = Trainer(
            devices=1,
            num_nodes=1,
            logger=wandb_logger,
        )

        # model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.test(model=model, dataloaders=data_module)
        test_f1_score = model.test_f1_score
        test_acc = model.test_acc_score

        wandb_logger.experiment.log({"cross_val_f1_score": best_score, 
                                    "cross_test_f1_score": test_f1_score,
                                    "cross_test_acc": test_acc})

        # clean up dataloader folder
        log.info("Cleaning up latent dataloader folder")
        data_folder = data_module.latent_dataloader.dataset_path
        os.system(f"rm -rf {data_folder}")




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

    

    input_dim = 2 if dataset == "asimow" else 1


    # load data
    dataset_dict = get_val_test_ids()


    wandb_logger = WandbLogger(log_model=True, project="asimow-predictive-quality", entity="tmdt-deep-learning")
    val_ids = dataset_dict["val_ids"]
    test_ids = dataset_dict["test_ids"]
    wandb_logger.experiment.log({"val_ids": str(val_ids), "test_ids": str(test_ids), "dataset-name": dataset})
    wandb_logger.log_hyperparams({"model_name": model_name})
    val_ids = dataset_dict['val_ids']
    test_ids = dataset_dict['test_ids']
    log.info(f"Val ids: {val_ids}")
    log.info(f"Test ids: {test_ids}")

    if dataset == "asimow":
        val_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in val_ids]
        test_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in test_ids]
        data_module = ASIMoWDataModule(task="reconstruction", batch_size=batch_size, n_cycles=1, val_data_ids=val_ids, test_data_ids=test_ids, use_augmentation=False)
        input_dim = 2
    else:
        raise ValueError(f"Invalid dataset name: {dataset}")
    data_module.setup(stage="fit")
    train_loader_size = len(data_module.train_ds)
    log.info(f"Loaded Data - Train dataset size: {train_loader_size}")

    if model_name == "VQ-VAE":
        model = VectorQuantizedVAE(
            wandb_logger=wandb_logger.experiment, input_dim=input_dim, hidden_dim=hidden_dim, num_embeddings=num_embeddings,
            embedding_dim=embedding_dim, n_resblocks=n_resblocks, learning_rate=learning_rate, decoder_type=decoder_type,  dropout_p=dropout_p
        )
    elif model_name == "VQ-VAE2":
        model = VectorQuantizedMultiTask(
            wandb_logger=wandb_logger.experiment, input_dim=input_dim, hidden_dim=hidden_dim, num_embeddings=num_embeddings,
            embedding_dim=embedding_dim, n_resblocks=n_resblocks, learning_rate=learning_rate, decoder_type=decoder_type, dropout_p=dropout_p
        )
    elif model_name == "VQ-VAE-Patch":
        model = VQVAEPatch(
            wandb_logger=wandb_logger.experiment, hidden_dim=hidden_dim, input_dim=input_dim, num_embeddings=num_embeddings,
            embedding_dim=embedding_dim, n_resblocks=n_resblocks, learning_rate=learning_rate, dropout_p=dropout_p, patch_size=patch_size
        )
    else:
        raise ValueError("Invalid model name")
    
    
    model_checkpoint_name = f"{model_name}-{dataset}-best"
    checkpoint_callback = ModelCheckpoint(dirpath='model_checkpoints/', monitor="val/loss", mode="min", filename=model_checkpoint_name)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.0001, patience=10, verbose=False, mode="min")
    
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        max_epochs=epochs,
        logger=wandb_logger, 
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
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.test(model=model, datamodule=data_module)
    
    classify_latent_space(latent_model=model, wandb_logger=wandb_logger, val_ids=val_ids, test_ids=test_ids, n_cycles=1, model_name=model_name, 
                            dataset=dataset, dataset_id=dataset_id, classification_model="MLP", learning_rate=learning_rate, clipping_value=clipping_value)
    
    wandb_logger.experiment.finish()

if __name__ == '__main__':
    wandb.require(experiment="service")
    os.environ["WANDB_DISABLE_SERVICE"] = "true"
    parser = argparse.ArgumentParser(description='Train VQ-VAE')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=25)
    parser.add_argument('--dataset', type=str, help='Dataset', default="asimow")
    parser.add_argument('--num-embeddings', type=int, help='Number of embeddings', default=256)
    parser.add_argument('--embedding-dim', type=int, help='Dimension of one embedding', default=32)
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension', default=512)
    parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--clipping-value', type=float, help='Gradient Clipping', default=0.7)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=512)
    parser.add_argument('--n-resblocks', type=int, help='Number of Residual Blocks', default=4)
    parser.add_argument('--patch-size', type=int, help='Patch size of the VQ-VAE Encoder', default=25)
    parser.add_argument('--dropout-p', type=float, help='Dropout probability', default=0.1)
    parser.add_argument('--model-name', type=str, help='Model name', default="VQ-VAE-Patch")
    parser.add_argument('--decoder-type', type=str, help='VQ-VAE Decoder Type', default="Conv")

    args = parser.parse_args()

    matplotlib.use('agg')
    
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision('medium')
    main(args)
