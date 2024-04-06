import pandas as pd
from tqdm import tqdm
import os
import gc  # Garbage collector for memory management

chunk_size = 100000000  # Adjust this as necessary

file_path = "raw_data.csv"
if not os.path.exists(file_path):
    print(f"File {file_path} not found. Please download it from the provided link in README.")
    exit(1)
chunks = pd.read_csv("raw_data.csv", chunksize=chunk_size)

for i, chunk in enumerate(tqdm(chunks)):
    chunk.to_pickle(f"raw_chunks/chunk_{i}.pkl")
    # Delete the chunk to free its memory
    del chunk
    # Explicitly call the garbage collector
    gc.collect()
    print(f"Processed chunk {i}")