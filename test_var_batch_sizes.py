import os
import matplotlib.pyplot as plt
import argparse
import torch

batch_sizes = [64, 128, 256, 512]

def get_file_names(directory):
    """
    Get all file names in the specified directory.
    :param directory: Path to the directory.
    :return: List of file names.
    """
    try:
        # List all entries in the directory
        entries = os.listdir(directory)

        # Filter out directories, keep only files
        file_names = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
        
        return file_names
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def main(args):
# Example usage
    
    directory_path = args.directory_path
    files = get_file_names(directory_path)
    print("Files in directory:", files)

    # exit(0)
    for model in files:
        print(f"Testing {model}")
        os.system(f"python anomaly_detection.py --model_path={directory_path + model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, help='Directory path', default="model_checkpoints/VQ-VAE-Patch/")

    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    main(args)