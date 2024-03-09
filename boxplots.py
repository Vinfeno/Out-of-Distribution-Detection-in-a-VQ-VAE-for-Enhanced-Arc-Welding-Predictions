import argparse
import logging as log
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataloader.asimow_dataloader import ASIMoWDataModule
from dataloader.utils import get_experiment_ids
from model.vq_vae_patch_embedd import VQVAEPatch
from plotting_utils import get_losses, print_loss_stats
from dataloader.utils import (
    get_experiment_ids,
    get_inv_vd_val_test_ids,
    get_inv_vs_val_test_ids,
    get_vs_val_test_ids,
    get_vd_val_test_ids,
)


def plot_vq_loss_distributions(model_path, image_path, data_split, title=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAEPatch.load_from_checkpoint(model_path)
    model = model.to(device)

    if title == None:
        title = "VQ Loss Distribution by Experiment"

    if data_split == "vs":
        val_ids = get_vs_val_test_ids()["val_ids"]
        test_ids = get_vs_val_test_ids()["test_ids"]
        test_ids_list = [f"{exp_id}_{run}" for exp_id, run in test_ids]
        experiment_ids = ()
        for ex_idx in [1, 2, 3]:
            experiment_ids += get_experiment_ids(ex_idx)
        experiment_ids = [exp_id for exp_id in experiment_ids if exp_id not in val_ids]
        loss_data = []  # Initialize loss_data as an empty list
        for experiment, run in experiment_ids:
            with open(
                f"dataloader_pickles/dataloader_experiment_{experiment}_run_{run}.pkl",
                "rb",
            ) as file:
                dataloader = pickle.load(file)
            vq_losses, _, _ = get_losses(dataloader=dataloader, model=model)

            # Store VQ loss data
            loss_data.extend(
                [
                    {"Experiment ID": f"{experiment}_{run}", "Value": vq_loss}
                    for vq_loss in vq_losses
                ]
            )

        # Convert list to DataFrame
        loss_df = pd.DataFrame(loss_data)
        loss_df = loss_df.set_index("Experiment ID")

    elif data_split == "vd":
        data_split = "vd"
    elif data_split == "vs-inf":
        data_split = "vs-inf"
    elif data_split == "vd-inf":
        data_split = "vd-inf"
    elif data_split == "ex":
        loss_data = []
        for ex_idx in [1, 2, 3]:
            experiment_ids = get_experiment_ids(ex_idx)

            for idx, (experiment, run) in enumerate(experiment_ids):
                with open(
                    f"dataloader_pickles/dataloader_experiment_{experiment}_run_{run}.pkl",
                    "rb",
                ) as file:
                    dataloader = pickle.load(file)
                    vq_losses, _, _ = get_losses(dataloader=dataloader, model=model)

                    # Store VQ loss data
                    loss_data.extend(
                        [
                            {"Experiment ID": f"{experiment}_{run}", "Value": vq_loss}
                            for vq_loss in vq_losses
                        ]
                    )
    else:
        raise ValueError(
            "Invalid data_split value. Expected 'vs', 'vd', 'vs-inf', or 'vd-inf'."
        )
    # Move Experiment IDs in test_ids_list to the end of the DataFrame
    plt.figure(figsize=(20, 10))
    loss_df.boxplot(column="Value", by="Experiment ID", ax=plt.gca(), rot=45)

    # Get the positions of the experiment IDs in test_ids_list
    test_ids_positions = [loss_df.index.get_loc(exp_id) for exp_id in test_ids_list]

    # Move the experiment IDs to the right side of the plot
    ax = plt.gca()
    for i, exp_id_pos in enumerate(test_ids_positions):
        ax.get_xticklabels()[exp_id_pos].set_horizontalalignment("right")

    plt.title(title)
    plt.tight_layout()

    plt.savefig(f"{image_path}.png", dpi=300)
    plt.show()
    # Initialize a DataFrame to store all the VQ loss data


def main(hparams):
    model_name = hparams.model_name
    checkpoint_name = hparams.checkpoint_name
    model_path = hparams.model_path + checkpoint_name
    image_path = hparams.image_path + checkpoint_name[:-5]
    data_split = hparams.data_split

    # Check if data_split is valid
    if data_split not in ["ex", "vs", "vd", "vs-inf", "vd-inf"]:
        raise ValueError(
            "Invalid data_split value. Expected 'ex', 'vs', 'vd', 'vs-inf', or 'vd-inf'."
        )

    title = hparams.title
    if title is None:
        title = checkpoint_name

    plot_vq_loss_distributions(model_path, image_path, hparams.data_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print and plots stats on embedding and reconstruction loss"
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        help="Checkpoint to analyze on.",
        default="VQ-VAE-Patch-asimow-vs-epochs=10-nEmb=16-beta=0.01.ckpt",
    )
    parser.add_argument(
        "--model-name", type=str, help="Just default", default="VQ-VAE-Patch"
    )
    parser.add_argument("--title", type=str, help="Table title", default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model",
        default="model_checkpoints/VQ-VAE-Patch/Series2/ParameterSplit/",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Save location for image",
        default="images/Boxplot/",
    )
    parser.add_argument(
        "--data-split", type=str, help="Data split condition", required=True
    )

args = parser.parse_args()
main(args)
