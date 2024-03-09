import argparse
from plotting_utils import plot_embedding_and_recon_loss

def main(hparams):
    plot_embedding_and_recon_loss(checkpoint_name=hparams.checkpoint_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print and plots stats on embedding and reconstruction loss')
    parser.add_argument('--checkpoint-name', type=str, help='Checkpoint to analyse on.')
    args = parser.parse_args()
    main(args)