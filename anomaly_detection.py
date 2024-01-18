import argparse
import os
import logging as log
import torch
import matplotlib
from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule
from dataloader.latentspace_dataloader import LatentPredDataModule
from dataloader.utils import get_val_test_ids
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from model.mlp import MLP
from model.gru import GRU
from model.classification_model import ClassificationLightningModule
import numpy as np
import tqdm.auto as tqdm


def print_training_input_shape(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    for i in range(len(batch)):
        log.info(f"Input {i} shape: {batch[i].shape}")
    

def classify_latent_space(latent_model: VectorQuantizedVAE | VQVAEPatch, logger: CSVLogger | WandbLogger, val_ids: list[DataSplitId], 
                          test_ids: list[DataSplitId], n_cycles: int, model_name: str, dataset: str,
                          classification_model: str, learning_rate: float, clipping_value: float):

    # Initialize a data module for latent space prediction
    data_module = LatentPredDataModule(latent_space_model=latent_model, model_name=f"{model_name}", val_data_ids=val_ids, test_data_ids=test_ids,
                                       n_cycles=n_cycles, task='classification', batch_size=128, model_id=f"{model_name}-{dataset}")
    # Print the shape of training data for verification
    print("_______________________")
    print("Shape:")
    print_training_input_shape(data_module)
    print("_______________________")
    # Calculate sequence length and input dimension for the classification model
    seq_len = n_cycles
    input_dim = int(latent_model.embedding_dim * latent_model.enc_out_len)

    # Select the classification model (MLP or GRU) based on the provided argument
    Model: type[MLP] | type[GRU]
    if classification_model == "MLP":
        Model = MLP
    elif classification_model == "GRU":
        Model = GRU
    else:
        raise ValueError(f"Invalid classification model name: {classification_model}")

    # Initialize the classification model with specified parameters
    model = Model(input_size=seq_len, in_dim=input_dim, hidden_sizes=128, dropout_p=0.1,
                  n_hidden_layers=4, output_size=2, learning_rate=learning_rate)

    # Set up checkpointing and early stopping based on F1 score
    model_checkpoint_name = f"VQ-VAE-{classification_model}-{dataset}-best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"model_checkpoints/VQ-VAE-{classification_model}/", monitor=f"val/f1_score", mode="max", filename=model_checkpoint_name)
    early_stop_callback = EarlyStopping(
        monitor=f"val/f1_score", min_delta=0.0001, patience=10, verbose=False, mode="max")

    # Initialize the PyTorch Lightning trainer with specified configurations
    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        devices=1,
        num_nodes=1,
        gradient_clip_val=clipping_value,
        check_val_every_n_epoch=1
    )

    # Train the model using the data module
    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Log the best F1 score and accuracy obtained during validation
    best_score = model.hyper_search_value
    best_acc_score = model.val_acc_score
    print(f"best score: {best_score}")
    print("------ Testing ------")

    # Reinitialize the trainer for testing
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger,
    )

    # Test the model using the test data
    trainer.test(model=model, dataloaders=data_module)
    test_f1_score = model.test_f1_score
    test_acc = model.test_acc_score

    # Log test metrics (F1 score and accuracy)
    logdict = {"val/mean_f1_score": best_score, 
               "val/mean_acc": best_acc_score,
               "test/mean_f1_score": test_f1_score,
               "test/mean_acc": test_acc}
    
    # Log metrics using the appropriate logger (CSV or Wandb)
    if isinstance(logger, CSVLogger):
        logger.experiment.log_metrics(logdict)
    else: 
        logger.experiment.log(logdict)
        logger.experiment.finish()

    # Clean up the data folder used by the data module
    log.info("Cleaning up latent dataloader folder")
    data_folder = data_module.latent_dataloader.dataset_path
    os.system(f"rm -rf {data_folder}")

def get_losses(dataloader: ASIMoWDataModule, model: VQVAEPatch) -> (np.array, np.array, np.array):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)
    recon_loss = []
    embedding_losses=[]
    perplexities = []
    model.eval()
    for batch in dataloader:
        input_data = batch[0].to(device)
        with torch.inference_mode():
            embedding_loss, x_hat, perplexity = model(input_data)
            recon_loss.append(np.mean((input_data.cpu().numpy() - x_hat.cpu().numpy())**2))
            embedding_losses.append(embedding_loss.cpu().numpy())
            perplexities.append(perplexity.cpu().numpy())
    return np.array(embedding_losses), np.array(recon_loss), np.array(perplexities)

def print_loss_stats(embed_loss: np.array, recon_loss: np.array, perplexity:np.array) -> None:
    print("Embedding loss:")
    print(f"Max: {np.max(embed_loss)}")
    print(f"Min: {np.min(embed_loss)}")
    print(f"Mean: {np.mean(embed_loss)}")
    print(f"Variance: {np.var(embed_loss)}")
    print("-------------------------------------------------------------")
    print("Reconstruction loss:")
    print(f"Max: {np.max(recon_loss)}")
    print(f"Min: {np.min(recon_loss)}")
    print(f"Mean: {np.mean(recon_loss)}")
    print(f"Variance: {np.var(recon_loss)}")
    print("-------------------------------------------------------------")
    corr_coeff = np.corrcoef(embed_loss, recon_loss)[0, 1]
    print("Pearson Correlation Coefficient: ", corr_coeff)
    print("-------------------------------------------------------------")
    print("Perplexity: ")
    print(f"Max: {np.max(perplexity)}")
    print(f"Min: {np.min(perplexity)}")
    print(f"Mean: {np.mean(perplexity)}")
    print(f"Variance: {np.var(perplexity)}")
    print("-------------------------------------------------------------\n")


def main(hparams):
    # Read hyperparameters from the provided hparams object
    batch_size = hparams.batch_size
    model_path = hparams.model_path

    # Repeat loading of dataset IDs and log them
    dataset_dict = get_val_test_ids()
    val_ids = dataset_dict["val_ids"]
    test_ids = dataset_dict["test_ids"]

    # Log the validation and test IDs
    val_ids = dataset_dict['val_ids']
    test_ids = dataset_dict['test_ids']
    log.info(f"Val ids: {val_ids}")
    log.info(f"Test ids: {test_ids}")
    
    ex3_ids = ()
    for e, w in val_ids:
        if e == 3:
            ex3_ids += ((e,w),)

    for e, w in test_ids:
        if e == 3:
             ex3_ids += ((e,w),)


    ex3_run_ids = ()
    for run in ex3_ids:
        ex3_run_ids += (run, )

    print(ex3_run_ids)

    ex_1_2_ids = ()
    for e, w in val_ids:
        if e != 3:
            ex_1_2_ids += ((e,w),)
    for e, w in test_ids:
        if e != 3:
            ex_1_2_ids += ((e,w),)
    
    ex3_datamodules = []
    # Setup data module specifically for the 'asimow' dataset
    ex3_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in ex3_ids]
    data_module_ex3 = ASIMoWDataModule(task="reconstruction", batch_size=batch_size, n_cycles=1, val_data_ids=ex3_ids, test_data_ids=ex3_ids)
    print("Loading ex3 runs...")
    for run_id in ex3_run_ids:
        run_id = [DataSplitId(experiment=run_id[0], welding_run=run_id[1])]
        ex3_datamodules.append(ASIMoWDataModule(task="reconstruction", batch_size=batch_size, n_cycles=1, val_data_ids=run_id, test_data_ids=run_id))
    ex_1_2_ids = [DataSplitId(experiment=e, welding_run=w) for e, w in ex_1_2_ids]
    data_module_ex1_2 = ASIMoWDataModule(task="reconstruction", batch_size=batch_size, n_cycles=1, val_data_ids=ex_1_2_ids, test_data_ids=ex_1_2_ids)

    # Prepare the data modules for training
    data_module_ex1_2.setup(stage="fit")
    data_module_ex3.setup(stage="fit")
    for i, dm in enumerate(ex3_datamodules):
        dm.setup(stage="fit")
    data_module_ex3.setup(stage="fit")
    data_module_ex1_2.setup(stage="fit")

    dataloader_lengths = [len(datamodule.train_ds) for datamodule in ex3_datamodules]
    train_loader_size_ex3 = len(data_module_ex3.train_ds)  # Get the size of the training dataset
    train_loader_size_ex1_2 = len(data_module_ex1_2.train_ds)  # Get the size of the training dataset
    for i in range(len(ex3_datamodules)):
        log.info(f"Loaded Data Ex 3, Run {i+1} - dataset size: {dataloader_lengths[i]}")
    log.info(f"Loaded Data Ex 3- Train dataset size: {train_loader_size_ex3}")
    log.info(f"Loaded Data Ex 1 2- Train dataset size: {train_loader_size_ex1_2}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAEPatch.load_from_checkpoint(model_path)
    model = model.to(device)


    # for datamodule in ex3_datamodules:
    #     datamodule.setup(stage="fit")
    data_module_ex3.setup(stage="fit")
    data_module_ex1_2.setup(stage="fit")

    # Get DataLoader
    train_loader_ex3 = data_module_ex3.train_dataloader()
    train_loader_ex1_2 = data_module_ex1_2.train_dataloader()
    for i, dm in enumerate(ex3_datamodules):
        ex3_datamodules[i] = dm.train_dataloader()



    print("#####\tExperiment 1 & 2\t#####")
    embedding_losses_ex_1_2, recon_loss_ex_12, perplexity = get_losses(train_loader_ex1_2, model)
    print_loss_stats(embedding_losses_ex_1_2, recon_loss_ex_12, perplexity)

    print("#####\tExeriment 3\t#####")
    embedding_losses_ex_3, recon_loss_ex_3, perplexity = get_losses(train_loader_ex3, model)
    print_loss_stats(embedding_losses_ex_3, recon_loss_ex_3, perplexity)

    for i, dm in enumerate(ex3_datamodules):
        print(f"#####\tExeriment 3, Run {i}\t#####")  
        embedding_losses_ex_3_run, recon_loss_ex_3_run, perplexity = get_losses(dm, model)
        print_loss_stats(embedding_losses_ex_3_run, recon_loss_ex_3_run, perplexity)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--model_path', type=str, help='Model path', default="model_checkpoints/VQ-VAE-Patch/VQ-VAE-Patch-asimow-best.ckpt")

    args = parser.parse_args()

    matplotlib.use('agg')
    
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision('medium')
    main(args)
