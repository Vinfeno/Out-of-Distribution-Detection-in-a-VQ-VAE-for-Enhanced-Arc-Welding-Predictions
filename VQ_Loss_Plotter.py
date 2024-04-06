import pickle
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms

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
        model = VQVAEPatch.load_from_checkpoint(model_path)
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

    def move_rows(self, df, test_ids):
        rows_to_move = df[df["Experiment ID"].isin(test_ids)]
        df = df[~df["Experiment ID"].isin(rows_to_move)]
        df = pd.concat([df, rows_to_move])
        return df

    def drop_and_move_rows(self, df, getter):
        unique_experiments = df[["Experiment ID"]].drop_duplicates().reset_index()
        test_ids = [f"{exp[0]}_{exp[1]}" for exp in getter()["test_ids"]]
        test_ids = [
            idx
            for idx in test_ids
            if idx in unique_experiments["Experiment ID"].to_list()
        ]
        return df, test_ids

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
        if data_split in ["ex", "ex-inv", "all"]:
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
        elif data_split in ["ex", "all"]:
            test_ids = df[df["Experiment ID"].str.startswith("3_")][
                "Experiment ID"
            ].unique()
        elif data_split == "ex-inv":
            test_ids = df[~df["Experiment ID"].str.startswith("3_")][
                "Experiment ID"
            ].unique()
        else:
            raise ValueError(f"Unknown data split: {data_split}")
        return df, test_ids

    def make_boxplot_all_data(
        self,
        title,
        data_split,
        image_path,
        file_name,
        model_path,
        figsize=(10, 10),
        rotation=45,
        custom_selection=None,
        amount=0,
        xlim=None,
        ylim=None,
    ):
        if data_split not in ["ex", "ex-inv", "vs", "vd", "vs-inv", "vd-inv"]:
            raise ValueError(f"Unknown data split: {data_split}")
        split_dict = self.split_data("all")
        loss_df = self.get_vq_loss_df(split_dict=split_dict, model_path=model_path)
        loss_df, test_ids = self.filter_val_data(loss_df, data_split)
        self.make_boxplot(
            df=loss_df,
            title=title,
            image_path=image_path,
            test_ids=test_ids,
            data_split=data_split,
            file_name=file_name,
            rotation=rotation,
            figsize=figsize,
            amount=amount,
            custom_selection=custom_selection,
            xlim=xlim,
            ylim=ylim,
        )

    def make_boxplot(
        self,
        df,
        title,
        image_path,
        test_ids,
        file_name,
        figsize,
        rotation,
        xlim=None,
        ylim=None,
        custom_selection=None,
        amount=0,
    ):
        # Move Experiment IDs in test_ids_list to the end of the DataFrame
        df["Experiment ID"] = df["Experiment ID"].apply(
            lambda x: "Train" if x not in test_ids else x
        )
        train = df[df["Experiment ID"] == "Train"]
        df.drop(df[df["Experiment ID"] == "Train"].index, inplace=True)
        df = pd.concat([train, df])
        if custom_selection is not None:
            runs_list = custom_selection
        elif amount == 0:
            runs_list = df["Experiment ID"].unique().tolist()
        else:
            runs_list = df["Experiment ID"].unique().tolist()[:amount]
        fig = plt.figure(figsize=figsize)

        df = df[df["Experiment ID"].isin(runs_list)]

        df["Color"] = "Other"

        # Now, update the 'Color' column where the condition matches, using .loc
        df = df.copy()
        df.loc[df["Experiment ID"].apply(lambda x: x not in test_ids), "Color"] = (
            "Train"
        )

        # Step 2: Define a custom palette
        palette = {"Train": "salmon", "Other": "lightblue"}
        order = ["Train"] + [
            eid for eid in df["Experiment ID"].unique() if eid != "Train"
        ]
        # Create the plot
        sns.set_theme(style="ticks")
        ax = sns.boxplot(
            data=df,
            x="Experiment ID",
            y="Value",
            hue="Color",  # Use the new column for coloring
            palette=palette,  # Apply the custom palette
            showfliers=False,
            order=order,
        )
        ax.set(xlabel="Welding Run ID", ylabel="VQ Loss", title=title)

        # Hide the legend if you don't want it, as it might not be meaningful in this context
        plt.legend([], [], frameon=False)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        else:
            lower_limit = float("inf")
            for unique_experiment in df["Experiment ID"].unique():
                experiment_quantile = df[df["Experiment ID"] == unique_experiment][
                    "Value"
                ].quantile(0.25)
                if experiment_quantile < lower_limit:
                    lower_limit = experiment_quantile
            lower_limit *= 0.8
            plt.gca().set_ylim(bottom=lower_limit)  # Set the lower limit of y-axis to 0
        if rotation != 0:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
                # Offset in pixels, adjust as needed
                offset = transforms.ScaledTranslation(5 / 72, 0, fig.dpi_scale_trans)
                # Apply transformation
                label.set_transform(label.get_transform() + offset)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(f"{image_path}{file_name}.png", dpi=400)

    def get_vq_loss_df(self, split_dict, model_path):

        model = VQVAEPatch.load_from_checkpoint(model_path)
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
        return loss_df
        # Initialize a DataFrame to store all the VQ loss data

    def boxplots(
        self,
        data_splits: list,
        epochs: list,
        embeddings: list,
        betas: list,
        figsize=(10, 10),
        image_path: str = "images/",
        title=None,
        file_name=None,
        rotation=45,
        auto_filenames=False,
        xlim=None,
        ylim=None,
        custom_selection=None,
        amount=0,
    ):
        """
        Possible values:
            data_splits=["ex", "ex-inv","vs","vd", "vs-inv", "vd-inv"],\n
            epochs=[10, 20, 30, 50],\n
            embeddings=[16, 64, 256],\n
            betas=[1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
        """
        total_iterations = len(data_splits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        # Check if data_split is valid
        for data_split in data_splits:
            split_dict = self.split_data(data_split)
            if data_split in ["ex", "ex-inv"]:
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
                        loss_df, test_ids = self.filter_val_data(loss_df, data_split)
                        # if title is None:
                        #     # title = f"VQ-Losses Split={data_split} Epoch={epoch} Beta={beta} Embeddings={embedding}"
                        if auto_filenames:
                            file_name = f"VQ-Losses-Split={data_split}Epochs={epoch}-nEmb={embedding}-beta={beta}"
                        print(file_name)
                        self.make_boxplot(
                            df=loss_df,
                            title=title,
                            image_path=image_path,
                            test_ids=test_ids,
                            figsize=figsize,
                            file_name=file_name,
                            rotation=rotation,
                            xlim=xlim,
                            ylim=ylim,
                            custom_selection=custom_selection,
                            amount=amount,
                        )
                        progress_bar.update(1)
        progress_bar.close()

    def make_threshold_plot(
        self,
        title,
        image_path,
        test_df,
        thresholds,
        figsize,
        file_name=None,
        amount=0,
        custom_selection: list = None,
        xlim=None,
        ylim=None,
        rotation=0,
    ) -> None:
        if file_name is None:
            file_name = title
        plt.figure(figsize=figsize)
        sns.set_palette("bright")
        if custom_selection is not None:
            runs_list = custom_selection
        elif amount == 0:
            runs_list = test_df["Experiment ID"].unique().tolist()
        else:
            runs_list = test_df["Experiment ID"].unique().tolist()[:amount]
        for run in runs_list:
            ax = sns.lineplot(
                test_df[test_df["Experiment ID"] == run].reset_index()["Value"],
                label=str(run),
            )
        ax.grid(True)
        # Add horizontal lines
        plt.axhline(
            y=thresholds["q1"], color="black", linestyle="--", label="Q1", alpha=0.5
        )
        plt.axhline(
            y=thresholds["q3"], color="black", linestyle="--", label="Q3", alpha=0.5
        )
        plt.axhline(
            y=thresholds["mean"], color="black", linestyle="-", label="Mean", alpha=0.5
        )
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        if rotation != 0:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
        plt.title(title)
        plt.xlabel("Batch")
        plt.ylabel("VQ Loss")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(image_path + file_name + ".png", dpi=500)

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
        figsize=(10, 10),
        title: str = None,
        image_path: str = "images/",
        file_name=None,
        amount=0,
        xlim=None,
        ylim=None,
    ):
        """
        Possible values (or anything if models were trained accordingly):
            data_splits=["ex","ex-inv", "vs","vd", "vs-inv", "vd-inv"],\n
            epochs=[10, 20, 30, 50],\n
            embeddings=[16, 64, 256],\n
            betas=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
        """

        auto_filenames = True if file_name is None else False
        total_iterations = len(data_splits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        # Check if data_split is valid
        for data_split in data_splits:
            split_dict = self.split_data(data_split)
            if data_split in ["ex", "ex-inv", "all"]:
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
                        if auto_filenames:
                            file_name = f"Thresholds-Split={data_split}-Epoch={epoch}-Beta={beta}-Embeddings={embedding}"
                        if title is None:
                            title = f"Thresholds Split={data_split}-Epoch={epoch}-Beta={beta}-Embeddings={embedding}"
                        self.make_threshold_plot(
                            title=title,
                            image_path=image_path + "/",
                            test_df=test_loss,
                            thresholds=thresholds,
                            figsize=figsize,
                            file_name=file_name,
                            amount=amount,
                            xlim=xlim,
                            ylim=ylim,
                        )
                        progress_bar.update(1)
        progress_bar.close()

    def load_original_dataloader(self):
        with open("dataloader_pickles/dataloader_experiment_1_run_3.pkl", "rb") as file:
            original_dl = pickle.load(file)
        return original_dl

    def get_data_range(self, original_dl):
        min_val, max_val = float("inf"), -float("inf")
        for data in original_dl:  # Assuming the dataloader yields batch data directly
            if isinstance(data, torch.Tensor):
                batch_min = data.min().item()
                batch_max = data.max().item()
            else:
                # If data is not a tensor, fallback to numpy's min/max
                batch_min = np.min(data)
                batch_max = np.max(data)
            min_val = min(min_val, batch_min)
            min_val = max(min_val, 0)
            max_val = max(max_val, batch_max)
        return min_val, max_val

    def get_noise_data(self):
        torch.manual_seed(42)
        original_dl = self.load_original_dataloader()
        min_val, max_val = self.get_data_range(original_dl)
        noise_data_loader = []
        for data in original_dl:  # Assuming it yields batches directly
            shape = data.shape
            # Ensure data type and device compatibility with original data
            dtype, device = data.dtype, data.device
            # Generating noise data within the original data's range
            # Note: torch.rand generates values in [0, 1). Scale and shift to match min_val and max_val
            noise_data = (
                torch.rand(size=shape, dtype=dtype, device=device) * (max_val - min_val)
                + min_val
            )
            noise_data_loader.append(noise_data)
        return noise_data_loader

    def test_on_noise(
        self,
        data_splits,
        epochs,
        embeddings,
        betas,
        image_path: str = "images/Random/",
        figsize=(10, 10),
    ):
        noise_data = self.get_noise_data()
        total_iterations = len(data_splits) * len(epochs) * len(embeddings) * len(betas)
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        # Check if data_split is valid
        for data_split in data_splits:
            split_dict = self.split_data(data_split)
            if data_split in ["ex", "ex-inv"]:
                split_type = "ExperimentSplit"
            elif data_split in ["vs", "vd", "vs-inv", "vd-inv"]:
                split_type = "ParameterSplit"
            else:
                raise ValueError(f"Unknown data split: {data_split}")
            for epoch in epochs:
                for embedding in embeddings:
                    for beta in betas:
                        model_path = f"model_checkpoints/VQ-VAE-Patch/Series2/ExperimentSplit/VQ-VAE-Patch-asimow-ex-split-epochs={epoch}-nEmb={embedding}-beta={beta}.ckpt"
                        model = VQVAEPatch.load_from_checkpoint(model_path)
                        loss_data = []
                        vq_losses_noise, _, _ = self.get_losses(
                            dataloader=noise_data, model=model
                        )

                        loss_data.extend(
                            [
                                {"Experiment ID": f"Random", "Value": loss}
                                for loss in vq_losses_noise
                            ]
                        )
                        noise_loss = pd.DataFrame(loss_data)
                        loss_df = pd.concat(
                            [
                                noise_loss,
                                self.get_vq_loss_df(
                                    split_dict=split_dict, model_path=model_path
                                ),
                            ]
                        )
                        test_id = ["Random"]
                        title = "Random "
                        self.make_boxplot(
                            df=loss_df,
                            title=title,
                            image_path=image_path,
                            test_ids=test_id,
                            data_split=data_split,
                            file_name=f"Random-Split={data_split}Epochs={epoch}-nEmb={embedding}-beta={beta}",
                            figsize=figsize,
                            rotation=0,
                        )
                        progress_bar.update(1)
        progress_bar.close()
