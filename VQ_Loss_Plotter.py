import pickle
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler

# from anomaly_detection import get_losses
from dataloader.asimow_dataloader import ASIMoWDataModule
from dataloader.utils import (
    get_experiment_ids,
    get_inv_vd_val_test_ids,
    get_inv_vs_val_test_ids,
    get_vd_val_test_ids,
    get_vs_val_test_ids,
)
from model.vq_vae_patch_embedd import VQVAEPatch
from tqdm.notebook import tqdm
from tqdm import tqdm


class VQ_Loss_Plotter:
    def __init__(
        self,
        model_name="VQ-VAE-Patch",
    ):
        self.model_name = model_name
        self.dataloaders = {}
        for ex_idx in [1, 2, 3]:
            ex = get_experiment_ids(ex_idx)
            for e, w in ex:
                with open(
                    f"dataloader_pickles/dataloader_experiment_{e}_run_{w}.pkl", "rb"
                ) as file:
                    self.dataloaders[(e, w)] = pickle.load(file)

    def get_loss_df(self, data_dict, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = VQVAEPatch.load_from_checkpoint(model_path)
        model = model.to(device)

        all_losses = []  # List to store all losses objects
        for (e, w), dataloader in data_dict.items():
            vq_losses, recon_losses, _ = self.get_losses(dataloader, model)
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
                "Correlation\nCoefficient": np.corrcoef(vq_losses, recon_losses)[0, 1],
            }
            losses = pd.Series(stats)
            all_losses.append(losses)

        loss_df = pd.DataFrame(all_losses)
        loss_df.set_index("Experiment ID", inplace=True)
        return loss_df

    def move_rows(self, df, ids_to_move):
        rows_to_move = df.loc[ids_to_move]
        df = df.drop(ids_to_move)
        df = pd.concat([df, rows_to_move])
        return df

    def drop_and_move_rows(self, df, getter):
        ids_to_move = [f"{exp[0]}_{exp[1]}" for exp in getter()["test_ids"]]
        ids_to_move = [idx for idx in ids_to_move if idx in df.index]
        df = self.move_rows(df, ids_to_move)
        return df, ids_to_move

    def make_heatmap(self, df, title, image_path, separate_position, data_split):
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
            image_path += "ExperimentSplit/"
            unique_experiments = sorted(
                set(run_id.split("_")[0] for run_id in df.index)
            )
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
        elif data_split in ["vs", "vd", "vs-inf", "vd-inf"]:
            image_path += "ParameterSplit/"
            ax.axhline(separate_position, color="blue", linestyle="-")
        else:
            raise ValueError("Unknown data split.")

        # Adjust the layout
        plt.tight_layout()
        plt.savefig(f"{image_path + title}.png", dpi=400)

    def get_losses(self, dataloader: ASIMoWDataModule, model: VQVAEPatch):
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

    def split_data(self, datasplit: str):
        if datasplit == "ex":
            selected_dataloaders = self.dataloaders
        elif datasplit == "vs":
            selected_dataloaders = self.remove_val_data(get_vs_val_test_ids)
        elif datasplit == "vd":
            selected_dataloaders = self.remove_val_data(get_vd_val_test_ids)
        elif datasplit == "vs-inf":
            selected_dataloaders = self.remove_val_data(get_inv_vs_val_test_ids)
        elif datasplit == "vd-inf":
            selected_dataloaders = self.remove_val_data(get_inv_vd_val_test_ids)
        else:
            raise ValueError("Unknown data split.")
        return selected_dataloaders

    def remove_val_data(self, getter):
        return {
            k: v for k, v in self.dataloaders.items() if k not in getter()["val_ids"]
        }

    def filter_val_data(self, df, data_split):
        if data_split == "vs":
            df, ids_to_move = self.drop_and_move_rows(df, get_vs_val_test_ids)
        elif data_split == "vd":
            df, ids_to_move = self.drop_and_move_rows(df, get_vd_val_test_ids)
        elif data_split == "vs-inv":
            df, ids_to_move = self.drop_and_move_rows(df, get_inv_vs_val_test_ids)
        elif data_split == "vd-inv":
            df, ids_to_move = self.drop_and_move_rows(df, get_inv_vd_val_test_ids)
        elif data_split == "ex":
            ids_to_move = None
        else:
            raise ValueError("Unknown data split.")
        return df, ids_to_move

    def heatmaps(
        self,
        datasplits: list,
        epochs: list,
        embeddings: list,
        betas: list,
        image_path: str,
    ):
        total_iterations = len(datasplits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")

        for datasplit in datasplits:
            split_dict = self.split_data(datasplit)
            if datasplit == "ex":
                split_type = "ExperimentSplit"
            elif datasplit in ["vs", "vd", "vs-inf", "vd-inf"]:
                split_type = "ParameterSplit"
            else:
                raise ValueError("Unknown data split.")

            for epoch in epochs:
                for embedding in embeddings:
                    for beta in betas:
                        model_path = f"model_checkpoints/VQ-VAE-Patch/Series2/{split_type}/VQ-VAE-Patch-asimow-{datasplit}-split-epochs={epoch}-nEmb={embedding}-beta={beta}.ckpt"
                        loss_df = self.get_loss_df(split_dict, model_path)
                        loss_df, rows_to_move = self.filter_val_data(loss_df, datasplit)
                        if rows_to_move is not None:
                            separate_position = len(loss_df) - len(rows_to_move)
                        else:
                            separate_position = len(loss_df)
                        title = f"Split={datasplit} Epoch={epoch} Beta={beta} Embeddings={embedding}"
                        self.make_heatmap(
                            df=loss_df,
                            title=title,
                            image_path=image_path,
                            separate_position=separate_position,
                            data_split=datasplit,
                        )

                        progress_bar.update(1)

        progress_bar.close()
