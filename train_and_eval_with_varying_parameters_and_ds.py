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
                        f'python3 train_recon_embed_selected_runs.py --parameter="{split}" --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="{split}-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
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


def eval_param_split_models(
    data_splits,
    epochs_list,
    num_embeddings,
    betas,
    num_param_models,
    current_param_evaled,
    total,
    total_evaled,
):
    for split in data_splits:
        for epochs in epochs_list:
            for beta in betas:
                for embeddings in num_embeddings:
                    os.system(
                        f'python3 novelty_detection_parameter_split.py --checkpoint-name="{split}-epochs={epochs}-nEmb={embeddings}-beta={beta}" --parameter="{split}"'
                    )
                    current_param_evaled += 1
                    total_evaled += 1
                    print_status(
                        current_param_evaled,
                        num_param_models,
                        total_evaled,
                        total,
                        "Eval parameter split models",
                    )
    return total_evaled


def train_ex_split_models(
    epochs_list,
    num_embeddings,
    betas,
    num_ex_models,
    current_ex_trained,
    total,
    total_trained,
):
    for epochs in epochs_list:
        for embeddings in num_embeddings:
            for beta in betas:
                os.system(
                    f'python3 train_recon_embed_ex_1_2_only.py --epochs={epochs} --num-embeddings={embeddings} --beta={beta} --checkpoint-name="ex-split-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
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


def eval_ex_split_models(
    epochs_list,
    num_embeddings,
    betas,
    num_ex_models,
    current_ex_evaled,
    total,
    total_evaled,
):
    for epochs in epochs_list:
        for embeddings in num_embeddings:
            for beta in betas:
                os.system(
                    f'python3 analyze_embedding_and_recon_loss.py --checkpoint-name="ex-split-epochs={epochs}-nEmb={embeddings}-beta={beta}"'
                )
                current_ex_evaled += 1
                total_evaled += 1
                print_status(
                    current_ex_evaled,
                    num_ex_models,
                    total_evaled,
                    total,
                    "Eval experiment split models",
                )
    return total_evaled


def main(hparams):
    data_splits = hparams.data_splits
    epochs_list = hparams.epochs
    num_embeddings = hparams.num_embeddings
    betas = hparams.betas
    train = hparams.train
    eval = hparams.eval
    param_split = hparams.param_split
    ex_split = hparams.ex_split

    if (not train and not eval) or (not param_split and not ex_split):
        print("Nothing to do.")

    num_param_models = (
        len(data_splits) * len(epochs_list) * len(num_embeddings) * len(betas)
    )
    num_ex_models = len(epochs_list) * len(num_embeddings) * len(betas)
    if not param_split and ex_split:
        total = num_ex_models
    elif not ex_split and param_split:
        total = num_param_models
    else:
        total = num_param_models + num_ex_models

    total_trained = 0
    total_evaled = 0
    current_param_trained = 0
    current_param_evaled = 0
    current_ex_trained = 0
    current_ex_evaled = 0

    if train:
        if param_split:
            total_trained = train_param_split_models(
                data_splits,
                epochs_list,
                num_embeddings,
                betas,
                num_param_models,
                current_param_trained,
                total,
                total_trained,
            )
        if ex_split:
            total_trained = train_ex_split_models(
                epochs_list,
                num_embeddings,
                betas,
                num_ex_models,
                current_ex_trained,
                total,
                total_trained,
            )
    if eval:
        if param_split:
            total_evaled = eval_param_split_models(
                data_splits,
                epochs_list,
                num_embeddings,
                betas,
                num_param_models,
                current_param_evaled,
                total,
                total_evaled,
            )
        if ex_split:
            total_evaled = eval_ex_split_models(
                epochs_list,
                num_embeddings,
                betas,
                num_ex_models,
                current_ex_evaled,
                total,
                total_evaled,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print and plots stats on embedding and reconstruction loss"
    )
    parser.add_argument("--train", action="store_true", help="Train models?")
    parser.add_argument(
        "--no-train", action="store_false", dest="train", help="Do not train models?"
    )
    parser.add_argument("--eval", action="store_true", help="Eval models?")
    parser.add_argument(
        "--no-eval", action="store_false", dest="eval", help="Do not eval models?"
    )
    parser.add_argument(
        "--param-split", action="store_true", help="Parameter split experiment?"
    )
    parser.add_argument(
        "--no-param-split",
        action="store_false",
        dest="param_split",
        help="No parameter split experiment?",
    )
    parser.add_argument(
        "--ex-split", action="store_true", help="Experiment split experiment?"
    )
    parser.add_argument(
        "--no-ex-split",
        action="store_false",
        dest="ex_split",
        help="No experiment split experiment?",
    )
    parser.add_argument(
        "--data-splits",
        nargs="+",  # '+' means "at least one"
        help="Splits to be trained and tested on.",
        default=["vs", "vd", "vs-inv", "vd-inv"],
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,  # Use 'int' if you want to ensure the values are integers
        help="Epoch values",
        default=[10, 20, 30],
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
        default=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
    )

    args = parser.parse_args()
    main(args)
