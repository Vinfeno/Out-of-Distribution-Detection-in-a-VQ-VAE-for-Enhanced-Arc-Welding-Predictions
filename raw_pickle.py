import pandas as pd
from tqdm import tqdm
import gc  # Garbage collector

chunk_size = 100000000  # Adjust this as necessary
chunks = pd.read_csv("raw_data.csv", chunksize=chunk_size)

for i, chunk in enumerate(tqdm(chunks)):
    chunk.to_pickle(f"raw_chunks/chunk_{i}.pkl")
    # Delete the chunk to free its memory
    del chunk
    # Explicitly call the garbage collector
    gc.collect()
