import argparse
from plotting_utils import plot_embedding_and_recon_loss


def main(hparams):
    checkpoint_name = hparams.checkpoint_name
    parameter = hparams.parameter
    if parameter == "vs":
        plot_embedding_and_recon_loss(
            checkpoint_name=checkpoint_name,
            experiments=[1, 2],
            runs=[1, 2, 3, 16, 20, 21],
        )
    elif parameter == "vd":
        plot_embedding_and_recon_loss(
            checkpoint_name=checkpoint_name,
            experiments=[1, 2],
            runs=[1, 2, 3, 4, 17, 18, 20, 21],
        )
    elif parameter == "vs-inv":
        plot_embedding_and_recon_loss(
            checkpoint_name=checkpoint_name,
            experiments=[1, 2],
            runs=[7, 8, 9, 16, 20, 21],
        )
    elif parameter == "vd-inv":
        plot_embedding_and_recon_loss(
            checkpoint_name=checkpoint_name,
            experiments=[1, 2],
            runs=[9, 4, 10, 19, 20, 21],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print and plots stats on embedding and reconstruction loss"
    )
    parser.add_argument("--checkpoint-name", type=str, help="Checkpoint to analyse on.")
    parser.add_argument(
        "--parameter", type=str, help="Parameter to split on (vs or vd)."
    )
    args = parser.parse_args()
    main(args)
