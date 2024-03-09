import argparse
from plotting_utils import print_loss_stats, get_losses
import os
import logging as log
import torch
from dataloader.asimow_dataloader import ASIMoWDataModule
from dataloader.utils import (
    get_experiment_ids,
    get_inv_vd_val_test_ids,
    get_inv_vs_val_test_ids,
    get_vs_val_test_ids,
    get_vd_val_test_ids,
)
from model.vq_vae_patch_embedd import VQVAEPatch
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_statistics(model_path, experiments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAEPatch.load_from_checkpoint(model_path)
    model = model.to(device)

    all_stats = []  # List to store all stats_series objects

    for ex_idx in experiments:
        ids = get_experiment_ids(ex_idx)

        for e, w in ids:
            with open(
                f"dataloader_pickles/dataloader_experiment_{e}_run_{w}.pkl", "rb"
            ) as file:
                dataloader = pickle.load(file)
                vq_losses, recon_losses, _ = get_losses(
                    dataloader=dataloader, model=model
                )
                stats = {
                    "Experiment ID": f"{e}_{w}",
                    "Embedding\nLoss Max": np.max(vq_losses),
                    "Embedding\nLoss Min": np.min(vq_losses),
                    "Embedding\nLoss Mean": np.mean(vq_losses),
                    "Embedding\nLoss Variance": np.var(vq_losses),
                    "Reconstruction\nLoss Max": np.max(recon_losses),
                    "Reconstruction\nLoss Min": np.min(recon_losses),
                    "Reconstruction\nLoss Mean": np.mean(recon_losses),
                    "Reconstruction\nLoss Variance": np.var(recon_losses),
                    "Correlation\nCoefficient": np.corrcoef(vq_losses, recon_losses)[
                        0, 1
                    ],
                }

                stats_series = pd.Series(stats)
                all_stats.append(stats_series)

    stats_df = pd.DataFrame(all_stats)
    stats_df.set_index("Experiment ID", inplace=True)
    return stats_df


def move_rows(df, ids_to_move):
    rows_to_move = df.loc[ids_to_move]
    df = df.drop(ids_to_move)
    df = pd.concat([df, rows_to_move])
    return df


def drop_and_move_rows(df, getter):
    indexes_to_drop = [f"{exp[0]}_{exp[1]}" for exp in getter()["val_ids"]]
    df.drop(indexes_to_drop, inplace=True, errors="ignore")
    ids_to_move = [f"{exp[0]}_{exp[1]}" for exp in getter()["test_ids"]]
    ids_to_move = [idx for idx in ids_to_move if idx in df.index]
    df = move_rows(df, ids_to_move)
    return df, ids_to_move


def make_heatmap(df, title, path, data_split):
    if data_split == "vs":
        df, ids_to_move = drop_and_move_rows(df, get_vs_val_test_ids)
    elif data_split == "vd":
        df, ids_to_move = drop_and_move_rows(df, get_vd_val_test_ids)
    elif data_split == "vs-inv":
        df, ids_to_move = drop_and_move_rows(df, get_inv_vs_val_test_ids)
    elif data_split == "vd-inv":
        df, ids_to_move = drop_and_move_rows(df, get_inv_vd_val_test_ids)
    elif data_split == "ex":
        pass
    else:
        raise ValueError("Unknown data split.")

    plt.figure(figsize=(13, 10))
    # Standardize columns individually
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(
        scaler.fit_transform(df), index=df.index, columns=df.columns
    )

    # Use a diverging color map, as we have standardized the data
    cmap = sns.diverging_palette(230, 20, as_cmap=True, s=100, l=40, n=9)

    # Create the heatmap
    ax = sns.heatmap(standardized_df, fmt="g", cmap=cmap, center=0)

    # Add titles and labels as needed
    plt.title(title, fontsize=24)
    plt.ylabel("Run ID", fontsize=18)
    plt.xlabel("Metrics", fontsize=18)
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12)

    # Add horizontal lines to separate different experiments based on Run ID
    if data_split == "ex":
        unique_experiments = sorted(set(run_id.split("_")[0] for run_id in df.index))
        for exp in unique_experiments[
            1:
        ]:  # Skip the first one as it starts from the top
            # Find the last occurrence of the previous experiment
            last_of_previous = max(
                i
                for i, run_id in enumerate(df.index)
                if run_id.startswith(f"{int(exp)-1}_")
            )
            ax.axhline(
                last_of_previous + 1, color="blue", linewidth=2, linestyle="-"
            )  # Adjust color and linewidth as needed
    else:
        separator_position = len(df) - len(ids_to_move)
        ax.axhline(separator_position, color="blue", linestyle="-")

    # Adjust the layout
    plt.tight_layout()
    plt.savefig(f"{path}.png", dpi=400)


def main(hparamns):
    # model_name = hparamns.model_name
    checkpoint_name = hparamns.checkpoint_name
    model_path = hparamns.model_path + checkpoint_name
    image_path = hparamns.image_path + checkpoint_name[:-5]
    title = hparamns.title
    data_split = hparamns.data_split
    if title is None:
        title = checkpoint_name

    if data_split in ["vs", "vd", "vs-inv", "vd-inv"]:
        df = get_statistics(model_path=model_path, experiments=[1, 2])
        make_heatmap(df, title, image_path, data_split=data_split)
    elif data_split == "ex":
        df = get_statistics(model_path=model_path, experiments=[1, 2, 3])
        make_heatmap(df, title, image_path, data_split=data_split)
    else:
        print("Nothing to do.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print and plots stats on embedding and reconstruction loss"
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        help="Checkpoint to analyse on.",
        default="VQ-VAE-Patch-asimow-vd-epochs=50-nEmb=16-beta=0.1.ckpt",
    )
    parser.add_argument(
        "--model-name", type=str, help="just default", default="VQ-VAE-Patch"
    )
    parser.add_argument(
        "--parameter", type=str, help="Parameter to split on (vs or vd)."
    )
    parser.add_argument("--title", type=str, help="Table title", default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model",
        default="model_checkpoints/VQ-VAE-Patch/",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Save location for image",
        default="images/",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        help="ex or param",
        default="",
    )

    args = parser.parse_args()
    main(args)
