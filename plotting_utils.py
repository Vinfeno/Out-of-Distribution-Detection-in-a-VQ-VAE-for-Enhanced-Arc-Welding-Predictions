import os
import logging as log
import torch
from dataloader.asimow_dataloader import ASIMoWDataModule
from dataloader.utils import get_experiment_ids
from model.vq_vae_patch_embedd import VQVAEPatch
import numpy as np
import matplotlib.pyplot as plt
import pickle


def print_training_input_shape(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    for i in range(len(batch)):
        log.info(f"Input {i} shape: {batch[i].shape}")


def get_losses(dataloader: ASIMoWDataModule, model: VQVAEPatch):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)
    recon_loss = []
    embedding_losses = []
    perplexities = []
    model.eval()
    for batch in dataloader:
        input_data = batch[0].to(device)
        with torch.inference_mode():
            embedding_loss, x_hat, perplexity = model(input_data)
            recon_loss.append(
                np.mean((input_data.cpu().numpy() - x_hat.cpu().numpy()) ** 2)
            )
            embedding_losses.append(embedding_loss.cpu().numpy())
            perplexities.append(perplexity.cpu().numpy())
    return np.array(embedding_losses), np.array(recon_loss), np.array(perplexities)


def print_loss_stats(
    embed_loss: np.array, recon_loss: np.array, perplexity: np.array
) -> None:
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


def plot_embedding_loss(
    dataloaders: list,
    model: VQVAEPatch,
    ids: tuple,
    directory: str,
    model_checkpoint: str,
    title: str,
) -> None:
    if title is None:
        title = model_checkpoint
    path = f"{directory}/{model_checkpoint}_embedding_losses_experiment_{ids[0][0]}.png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    runs = [w for _, w in ids]
    losses = []
    for w, dl in zip(runs, dataloaders):
        emb_loss, _, _ = get_losses(dl, model)
        losses.append((w, emb_loss))

    plt.figure(figsize=(32, 6))
    for run in losses:
        plt.plot(run[1], label=str(run[0]))
    plt.title(f"{title}\n Embedding Loss")
    plt.xlabel("Batch")
    plt.ylabel("Embedding Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_recon_loss(
    dataloaders: list,
    model: VQVAEPatch,
    ids: tuple,
    directory: str,
    model_checkpoint: str,
    title: str,
) -> None:
    if title is None:
        title = model_checkpoint
    path = f"{directory}/{model_checkpoint}_reconstruction_losses_experiment_{ids[0][0]}.png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    runs = [w for _, w in ids]
    losses = []
    for w, dl in zip(runs, dataloaders):
        _, recon_loss, _ = get_losses(dl, model)
        losses.append((w, recon_loss))

    plt.figure(figsize=(32, 6))
    for run in losses:
        plt.plot(run[1], label=str(run[0]))
    plt.xlabel("Batch")
    plt.ylabel("Reconstruction Loss")
    plt.title(f"{title}\n Reconstruction Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_embedding_and_recon_loss(
    checkpoint_name: str,
    experiments: list = None,
    runs: list = None,
    model_name: str = "VQ-VAE-Patch",
    title: str = None,
):
    model_path = (
        f"model_checkpoints/{model_name}/{model_name}-asimow-{checkpoint_name}.ckpt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAEPatch.load_from_checkpoint(model_path)
    model = model.to(device)

    if experiments == None:
        experiments = [1, 2, 3]

    for ex_idx in experiments:
        dataloaders = []
        if runs == None:
            ex = get_experiment_ids(ex_idx)
        else:
            ex = []
            for _, w in get_experiment_ids(ex_idx):
                if w in runs:
                    ex.append((ex_idx, w))
            ex = tuple(ex)

        for e, w in ex:
            with open(
                f"dataloader_pickles/dataloader_experiment_{e}_run_{w}.pkl", "rb"
            ) as file:
                dataloaders.append(pickle.load(file))
        plot_embedding_loss(
            dataloaders=dataloaders,
            model=model,
            ids=ex,
            directory=f"images/{checkpoint_name}/",
            model_checkpoint=checkpoint_name,
            title=title,
        )
        plot_recon_loss(
            dataloaders=dataloaders,
            model=model,
            ids=ex,
            directory=f"images/{checkpoint_name}/",
            model_checkpoint=checkpoint_name,
            title=title,
        )
