import argparse
import os
from tqdm import tqdm


def print_status(type_count, type_sum, total_current, total_sum, process):
    print("#################################")
    print(
        f"{process}: ({type_count}/{type_sum})\t Total: ({total_current}/{total_sum})"
    )
    print("#################################")


def heatmap_param_split_models(
    data_splits,
    epochs_list,
    num_embeddings,
    betas,
    num_param_models,
    current_param_evaled,
    total,
    total_evaled,
    model_path,
    image_path,
):
    # Create a progress bar for the outermost loop
    with tqdm(
        total=num_param_models, desc="Processing parameter split models"
    ) as pbar_outer:
        for split in data_splits:
            for epochs in epochs_list:
                for beta in betas:
                    for embeddings in num_embeddings:
                        os.system(
                            f'python3 heatmap_losses.py --checkpoint-name="VQ-VAE-Patch-asimow-{split}-epochs={epochs}-nEmb={embeddings}-beta={beta}.ckpt" --parameter="{split}" --model-path="{model_path}" --title="Data Split: "{split}" Epochs={epochs} Beta={beta} Embeddings={embeddings}" --image-path={image_path} --data-split="{split}"'
                        )
                        current_param_evaled += 1
                        total_evaled += 1

                        # Update progress for each iteration
                        pbar_outer.update(1)
                        pbar_outer.set_postfix_str(
                            f"Evaluated: {current_param_evaled}/{num_param_models}, Total: {total_evaled}/{total}"
                        )
    return total_evaled


def heatmap_ex_split_models(
    epochs_list,
    num_embeddings,
    betas,
    num_ex_models,
    current_ex_evaled,
    total,
    total_evaled,
    model_path,
    image_path,
):
    # Calculate the total iterations for the progress bar
    total_iterations = len(epochs_list) * len(num_embeddings) * len(betas)

    # Initialize the progress bar
    with tqdm(
        total=total_iterations, desc="Processing experiment split models"
    ) as pbar:
        for epochs in epochs_list:
            for embeddings in num_embeddings:
                for beta in betas:
                    os.system(
                        f'python3 heatmap_losses.py --checkpoint-name="VQ-VAE-Patch-asimow-ex-split-epochs={epochs}-nEmb={embeddings}-beta={beta}.ckpt" --model-path="{model_path}" --title="Epoch={epochs} Beta={beta} Embeddings={embeddings}" --image-path={image_path} --data-split="ex"'
                    )
                    current_ex_evaled += 1
                    total_evaled += 1
                    # Update the progress bar after each command execution
                    pbar.update(1)
                    # Optionally, you can set a postfix to show additional information
                    pbar.set_postfix_str(
                        f"Evaluated: {current_ex_evaled}/{num_ex_models}, Total Evaluated: {total_evaled}/{total}"
                    )

    return total_evaled


def main(hparams):
    data_splits = hparams.data_splits
    epochs_list = hparams.epochs
    num_embeddings = hparams.num_embeddings
    betas = hparams.betas
    param_split = hparams.param_split
    ex_split = hparams.ex_split
    parameter_model_path = hparams.parameter_model_path
    experiment_model_path = hparams.experiment_model_path
    parameter_image_path = hparams.parameter_image_path
    experiment_image_path = hparams.experiment_image_path

    if not param_split and not ex_split:
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

    total_evaled = 0
    current_param_evaled = 0
    current_ex_evaled = 0
    if param_split:
        total_evaled = heatmap_param_split_models(
            data_splits=data_splits,
            epochs_list=epochs_list,
            num_embeddings=num_embeddings,
            betas=betas,
            num_param_models=num_param_models,
            current_param_evaled=current_param_evaled,
            total=total,
            total_evaled=total_evaled,
            model_path=parameter_model_path,
            image_path=parameter_image_path,
        )
    if ex_split:
        total_evaled = heatmap_ex_split_models(
            epochs_list=epochs_list,
            num_embeddings=num_embeddings,
            betas=betas,
            num_ex_models=num_ex_models,
            current_ex_evaled=current_ex_evaled,
            total=total,
            total_evaled=total_evaled,
            model_path=experiment_model_path,
            image_path=experiment_image_path,
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
        default=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
    )
    parser.add_argument(
        "--parameter-model-path",
        type=str,
        help="Path to model",
        default="model_checkpoints/VQ-VAE-Patch/Series2/ParameterSplit/",
    )
    parser.add_argument(
        "--experiment-model-path",
        type=str,
        help="Path to model",
        default="model_checkpoints/VQ-VAE-Patch/Series2/ExperimentSplit/",
    )
    parser.add_argument(
        "--parameter-image-path",
        type=str,
        help="Save location for image",
        default="images/Heatmap/ParameterSplit/",
    )
    parser.add_argument(
        "--experiment-image-path",
        type=str,
        help="Save location for image",
        default="images/Heatmap/ExperimentSplit/",
    )

    args = parser.parse_args()
    main(args)
