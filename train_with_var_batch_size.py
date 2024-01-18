import os

batch_sizes = [64, 128, 256, 512]

for batch_size in batch_sizes:
    os.system(f"python train_recon_embed_ex_1_2_only.py --epochs=10 --batch-size={batch_size} --model-name=Var_Batch_Size")


