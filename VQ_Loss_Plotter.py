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
        # loss_df.set_index("Experiment ID", inplace=True)
        return loss_df

    def move_rows(self, df, ids_to_move):
        rows_to_move = df[df["Experiment ID"].isin(ids_to_move)]
        df = df[~df["Experiment ID"].isin(rows_to_move)]
        df = pd.concat([df, rows_to_move])
        return df

    def drop_and_move_rows(self, df, getter):
        unique_experiments = df[["Experiment ID"]].drop_duplicates().reset_index()
        ids_to_move = [f"{exp[0]}_{exp[1]}" for exp in getter()["test_ids"]]
        ids_to_move = [
            idx
            for idx in ids_to_move
            if idx in unique_experiments["Experiment ID"].to_list()
        ]
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
                set(run_id.split("_")[0] for run_id in df["Experiment ID"].to_list())
            )
            for exp in unique_experiments[
                1:
            ]:  # Skip the first one as it starts from the top
                # Find the last occurrence of the previous experiment
                last_of_previous = max(
                    i
                    for i, run_id in enumerate(df["Experiment ID"].to_list())
                    if run_id.startswith(f"{int(exp)-1}_")
                )
                ax.axhline(
                    last_of_previous + 1, color="blue", linewidth=2, linestyle="-"
                )  # Adjust color and linewidth as needed
        elif data_split in ["vs", "vd", "vs-inv", "vd-inv"]:
            image_path += "ParameterSplit/"
            ax.axhline(separate_position, color="blue", linestyle="-")
        else:
            raise ValueError(f"Unknown data split: {data_split}")

        # Adjust the layout
        plt.tight_layout()
        plt.savefig(f"{image_path + title}.png", dpi=400)
        plt.close()

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

    def split_data(self, data_split: str):
        if data_split == "ex":
            selected_dataloaders = self.dataloaders
        elif data_split == "vs":
            selected_dataloaders = self.remove_val_data(get_vs_val_test_ids)
        elif data_split == "vd":
            selected_dataloaders = self.remove_val_data(get_vd_val_test_ids)
        elif data_split == "vs-inv":
            selected_dataloaders = self.remove_val_data(get_inv_vs_val_test_ids)
        elif data_split == "vd-inv":
            selected_dataloaders = self.remove_val_data(get_inv_vd_val_test_ids)
        else:
            raise ValueError(f"Unknown data split: {data_split}")
        return selected_dataloaders

    def remove_val_data(self, getter):
        return {
            k: v for k, v in self.dataloaders.items() if k not in getter()["val_ids"]
        }

    def filter_val_data(self, df, data_split):
        if data_split == "vs":
            df, test_ids = self.drop_and_move_rows(df, get_vs_val_test_ids)
        elif data_split == "vd":
            df, test_ids = self.drop_and_move_rows(df, get_vd_val_test_ids)
        elif data_split == "vs-inv":
            df, test_ids = self.drop_and_move_rows(df, get_inv_vs_val_test_ids)
        elif data_split == "vd-inv":
            df, test_ids = self.drop_and_move_rows(df, get_inv_vd_val_test_ids)
        elif data_split == "ex":
            test_ids = df[df["Experiment ID"].str.startswith("3_")][
                "Experiment ID"
            ].unique()
        else:
            raise ValueError(f"Unknown data split: {data_split}")
        return df, test_ids

    def heatmaps(
        self,
        data_splits: list,
        epochs: list,
        embeddings: list,
        betas: list,
        image_path: str = "images/",
    ):
        """
        Possible values:
            data_splits=["ex","vs","vd", "vs-inv", "vd-inv"],\n
            epochs=[10, 20, 30, 50],\n
            embeddings=[16, 64, 256],\n
            betas=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
        """
        total_iterations = len(data_splits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")

        for data_split in data_splits:
            split_dict = self.split_data(data_split)
            if data_split == "ex":
                split_type = "ExperimentSplit"
            elif data_split in ["vs", "vd", "vs-inv", "vd-inv"]:
                split_type = "ParameterSplit"
            else:
                raise ValueError(f"Unknown data split: {data_split}")

            for epoch in epochs:
                for embedding in embeddings:
                    for beta in betas:
                        model_path = f"model_checkpoints/VQ-VAE-Patch/Series2/{split_type}/VQ-VAE-Patch-asimow-{data_split}-split-epochs={epoch}-nEmb={embedding}-beta={beta}.ckpt"
                        loss_df = self.get_loss_df(split_dict, model_path)
                        loss_df, rows_to_move = self.filter_val_data(
                            loss_df, data_split
                        )
                        if rows_to_move is not None:
                            separate_position = len(loss_df) - len(rows_to_move)
                        else:
                            separate_position = len(loss_df)
                        title = f"Split={data_split} Epoch={epoch} Beta={beta} Embeddings={embedding}"
                        self.make_heatmap(
                            df=loss_df,
                            title=title,
                            image_path=image_path,
                            separate_position=separate_position,
                            data_split=data_split,
                        )

                        progress_bar.update(1)

        progress_bar.close()

    def make_boxplot(self, df, title, image_path, data_split, train_ids):
        # Move Experiment IDs in test_ids_list to the end of the DataFrame
        plt.figure(figsize=(20, 10))
        df.boxplot(column="Value", by="Experiment ID", ax=plt.gca(), rot=45)

        # Highlight experiment IDs in train_ids
        ax = plt.gca()
        unique_experiments = df[["Experiment ID"]].drop_duplicates().reset_index()
        if data_split == "ex":
            for exp_id in unique_experiments["Experiment ID"]:
                if exp_id.startswith("3_"):
                    exp_id_pos = unique_experiments[
                        unique_experiments["Experiment ID"] == exp_id
                    ].index[0]
                    ax.get_xticklabels()[exp_id_pos].set_color("red")
        else:
            for exp_id in train_ids:
                exp_id_pos = unique_experiments[
                    unique_experiments["Experiment ID"] == exp_id
                ].index[0]
                ax.get_xticklabels()[exp_id_pos].set_color("red")

        plt.title(title)
        plt.tight_layout()

        plt.savefig(f"{image_path}/{title}.png", dpi=300)

        split_type = "ExperimentSplit" if data_split == "ex" else "ParameterSplit"

        plt.title(title)
        plt.tight_layout()

        plt.savefig(f"{image_path + split_type}/{title}.png", dpi=300)

    def get_vq_loss_df(self, split_dict, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = VQVAEPatch.load_from_checkpoint(model_path)
        model = model.to(device)
        loss_data = []
        for experiment, run in split_dict.keys():
            with open(
                f"dataloader_pickles/dataloader_experiment_{experiment}_run_{run}.pkl",
                "rb",
            ) as file:
                dataloader = pickle.load(file)

                vq_losses, _, _ = self.get_losses(dataloader=dataloader, model=model)

                # Store VQ loss data
                loss_data.extend(
                    [
                        {"Experiment ID": f"{experiment}_{run}", "Value": vq_loss}
                        for vq_loss in vq_losses
                    ]
                )

            # Convert list to DataFrame
        loss_df = pd.DataFrame(loss_data)
        # loss_df = loss_df.sort_values(
        #     by="Experiment ID", ascending=True
        # )  # Sort by "Experiment ID"

        return loss_df
        # Initialize a DataFrame to store all the VQ loss data

    def boxplots(
        self,
        data_splits: list,
        epochs: list,
        embeddings: list,
        betas: list,
        image_path: str = "images/",
    ):
        """
        Possible values:
            data_splits=["ex","vs","vd", "vs-inv", "vd-inv"],\n
            epochs=[10, 20, 30, 50],\n
            embeddings=[16, 64, 256],\n
            betas=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
        """
        total_iterations = len(data_splits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        # Check if data_split is valid
        for data_split in data_splits:
            split_dict = self.split_data(data_split)
            if data_split == "ex":
                split_type = "ExperimentSplit"
            elif data_split in ["vs", "vd", "vs-inv", "vd-inv"]:
                split_type = "ParameterSplit"
            else:
                raise ValueError(f"Unknown data split: {data_split}")

            for epoch in epochs:
                for embedding in embeddings:
                    for beta in betas:
                        model_path = f"model_checkpoints/VQ-VAE-Patch/Series2/{split_type}/VQ-VAE-Patch-asimow-{data_split}-split-epochs={epoch}-nEmb={embedding}-beta={beta}.ckpt"
                        loss_df = self.get_vq_loss_df(
                            split_dict=split_dict, model_path=model_path
                        )
                        loss_df, train_ids = self.filter_val_data(loss_df, data_split)
                        title = f"VQ-Losses Split={data_split} Epoch={epoch} Beta={beta} Embeddings={embedding}"
                        self.make_boxplot(
                            df=loss_df,
                            title=title,
                            image_path=image_path,
                            train_ids=train_ids,
                            data_split=data_split,
                        )
                        progress_bar.update(1)
        progress_bar.close()

    def make_threshold_plot(
        self,
        title,
        image_path,
        data_split,
        test_df,
        thresholds,
    ) -> None:
        plt.figure(figsize=(32, 6))
        for run in test_df["Experiment ID"].unique().tolist():
            plt.plot(
                test_df[test_df["Experiment ID"] == run].reset_index()["Value"],
                label=str(run),
            )

        # Add horizontal lines
        plt.axhline(y=thresholds["q1"], color="r", linestyle="--", label="Q1")
        plt.axhline(y=thresholds["q3"], color="r", linestyle="--", label="Q3")
        plt.axhline(y=thresholds["mean"], color="r", linestyle="-", label="Mean")
        # plt.axhline(y=thresholds["p5"], color="r", linestyle=":", label="P5")
        # plt.axhline(y=thresholds["p95"], color="r", linestyle=":", label="P95")

        plt.title(f"{title}\n Embedding Loss")
        plt.xlabel("Batch")
        plt.ylabel("VQ Loss")
        plt.legend()
        plt.savefig(image_path + title + ".png", dpi=400)
        plt.close()

    def get_thresholds(self, loss_df: pd.DataFrame):
        return {
            "mean": loss_df["Value"].mean(),
            "q1": loss_df["Value"].quantile(0.25),
            "q3": loss_df["Value"].quantile(0.75),
            "p5": loss_df["Value"].quantile(0.05),
            "p95": loss_df["Value"].quantile(0.95),
        }

    def split_train_test(self, df, data_split):
        train_df, test_ids = self.filter_val_data(df, data_split)
        test_df = train_df[train_df["Experiment ID"].isin(test_ids)]
        train_df = train_df[~train_df["Experiment ID"].isin(test_ids)]
        return train_df, test_df

    def thresholds(
        self,
        data_splits: list,
        epochs: list,
        embeddings: list,
        betas: list,
        image_path: str = "images/",
    ):
        """
        Possible values:
            data_splits=["ex","vs","vd", "vs-inv", "vd-inv"],\n
            epochs=[10, 20, 30, 50],\n
            embeddings=[16, 64, 256],\n
            betas=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
        """
        total_iterations = len(data_splits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        # Check if data_split is valid
        for data_split in data_splits:
            split_dict = self.split_data(data_split)
            if data_split == "ex":
                split_type = "ExperimentSplit"
            elif data_split in ["vs", "vd", "vs-inv", "vd-inv"]:
                split_type = "ParameterSplit"
            else:
                raise ValueError(f"Unknown data split: {data_split}")

            for epoch in epochs:
                for embedding in embeddings:
                    for beta in betas:
                        model_path = f"model_checkpoints/VQ-VAE-Patch/Series2/{split_type}/VQ-VAE-Patch-asimow-{data_split}-split-epochs={epoch}-nEmb={embedding}-beta={beta}.ckpt"
                        loss_df = self.get_vq_loss_df(
                            split_dict=split_dict, model_path=model_path
                        )
                        train_loss, test_loss = self.split_train_test(
                            df=loss_df, data_split=data_split
                        )
                        thresholds = self.get_thresholds(train_loss)
                        title = f"Thresholds\nSplit={data_split} Epoch={epoch} Beta={beta} Embeddings={embedding}"
                        self.make_threshold_plot(
                            title=title,
                            image_path=image_path,
                            data_split=data_split,
                            test_df=test_loss,
                            thresholds=thresholds,
                        )
                        progress_bar.update(1)
        progress_bar.close()
