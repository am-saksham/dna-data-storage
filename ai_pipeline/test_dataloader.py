import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time

class DNADataset(Dataset):
    def __init__(self, parquet_path):
        # PyArrow allows pandas to load massive parquets instantly
        print("Loading Parquet into RAM...")
        self.df = pd.read_parquet(parquet_path, engine='pyarrow')
        print(f"Loaded {len(self.df)} sequences.")
        
        # In a real model, we would tokenize 'A','C','G','T' into integers here.
        # For validation, we just return the strings.
        self.noisy = self.df['noisy_dna'].values
        self.clean = self.df['clean_dna'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

def test_speed():
    dataset = DNADataset("training_data.parquet")
    
    # DataLoader handles parallel batching
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)
    
    print("Testing PyTorch iteration speed...")
    start_time = time.time()
    
    batches_processed = 0
    for noisy_batch, clean_batch in dataloader:
        batches_processed += 1
        # In real life, move to GPU and pass to Transformer here
        
    elapsed = time.time() - start_time
    print(f"Processed {batches_processed} batches (100,000 sequences) in {elapsed:.4f} seconds.")
    print(f"Throughput: {len(dataset) / elapsed:,.0f} sequences per second!")

if __name__ == "__main__":
    test_speed()
