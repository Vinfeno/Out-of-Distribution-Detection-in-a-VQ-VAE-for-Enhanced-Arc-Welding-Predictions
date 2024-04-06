import argparse
import os


def print_status(type_count, type_sum, total_current, total_sum, process):
    print("#################################")
    print(
        f"{process}: ({type_count}/{type_sum})\t Total: ({total_current}/{total_sum})"
    )
    print("#################################")


def train_param_split_models(
    data_splits,
    epochs_list,
    num_embeddings,
    betas,
    num_param_models,
    current_param_trained,
    total,
    total_trained,
):
    for split in data_splits:
        for epochs in epochs_list:
            for beta in betas:
                for embeddings in num_embeddings:
                    os.system(
                        f'python3 train_recon_embed_selected_runs.py --parameter="{split}" --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="{split}-split-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
                    )
                    current_param_trained += 1
                    total_trained += 1
                    print_status(
                        current_param_trained,
                        num_param_models,
                        total_trained,
                        total,
                        "Train parameter split models",
                    )

    return total_trained


def train_ex_split_models(
    epochs_list,
    num_embeddings,
    betas,
    num_ex_models,
    current_ex_trained,
    total,
    total_trained,
    data_splits=["ex", "ex-inv"],
):
    for data_split in data_splits:
        for epochs in epochs_list:
            for embeddings in num_embeddings:
                for beta in betas:
                    os.system(
                        f'python3 train_vqvae_experiment_split.py --data-split={data_split} --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="{data_split}-split-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
                    )
                    current_ex_trained += 1
                    total_trained += 1
                    print_status(
                        current_ex_trained,
                        num_ex_models,
                        total_trained,
                        total,
                        "Train experiment split models",
                    )
    return total_trained


def main(hparams):
    epochs_list = hparams.epochs
    num_embeddings = hparams.num_embeddings
    betas = hparams.betas
    ex_splits = hparams.ex_splits
    param_splits = hparams.param_splits
    ex = hparams.ex
    param = hparams.param

    num_param_models = (
        (len(param_splits) + len(ex_splits))
        * len(epochs_list)
        * len(num_embeddings)
        * len(betas)
    )
    num_ex_models = len(epochs_list) * len(num_embeddings) * len(betas)
    if not param and ex:
        total = num_ex_models
    elif not ex and param:
        total = num_param_models
    else:
        total = num_param_models + num_ex_models

    total_trained = 0
    current_param_trained = 0
    current_ex_trained = 0

    if param:
        total_trained = train_param_split_models(
            param_splits,
            epochs_list,
            num_embeddings,
            betas,
            num_param_models,
            current_param_trained,
            total,
            total_trained,
        )
    if ex:
        total_trained = train_ex_split_models(
            epochs_list,
            num_embeddings,
            betas,
            num_ex_models,
            current_ex_trained,
            total,
            total_trained,
            data_splits=ex_splits,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print and plots stats on embedding and reconstruction loss"
    )
    parser.add_argument(
        "--param", action="store_true", help="Parameter split experiment?"
    )
    parser.add_argument(
        "--no-param",
        action="store_false",
        dest="param_split",
        help="No parameter split experiment?",
    )
    parser.add_argument(
        "--ex", action="store_true", help="Experiment split experiment?"
    )
    parser.add_argument(
        "--no-ex",
        action="store_false",
        dest="ex_split",
        help="No experiment split experiment?",
    )
    parser.add_argument(
        "--param-splits",
        nargs="+",  # '+' means "at least one"
        help="Splits to be trained and tested on.",
        default=["vs", "vd", "vs-inv", "vd-inv"],
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,  # Use 'int' if you want to ensure the values are integers
        help="Epoch values",
        default=[10, 20, 30, 50],
    )
    parser.add_argument(
        "--num_embeddings",
        nargs="+",
        type=int,  # Use 'int' for integers
        help="Number of embedding values",
        default=[16, 64, 256],
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,  # Use 'float' if these are floating-point numbers
        help="Beta values",
        default=[1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
    )
    parser.add_argument(
        "--ex-splits",
        nargs="+",
        type=str,
        help='ex-splits: ["ex", "ex-inv"]',
        default=["ex", "ex-inv"],
    )

    args = parser.parse_args()
    main(args)
