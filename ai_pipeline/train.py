import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import DNADenoiserTransformer, DNAVocabulary

# 1. Dataset Loader & Tokenization
class DNADataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path} into RAM...")
        self.df = pd.read_parquet(parquet_path, engine='pyarrow')
        self.noisy = self.df['noisy_dna'].values
        self.clean = self.df['clean_dna'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

def collate_fn(batch):
    """
    Transforms a batch of raw DNA strings into padded integer tensors.
    """
    vocab = DNAVocabulary()
    noisy_strs, clean_strs = zip(*batch)
    
    # Encode strings to integer lists
    noisy_encoded = [vocab.encode(seq) for seq in noisy_strs]
    clean_encoded = [vocab.encode(seq) for seq in clean_strs]
    
    # Dynamic padding saves VRAM!
    max_noisy = max(len(s) for s in noisy_encoded)
    max_clean = max(len(s) for s in clean_encoded)
    
    # Pad with 0 (<PAD>)
    noisy_padded = [s + [vocab.pad_token] * (max_noisy - len(s)) for s in noisy_encoded]
    clean_padded = [s + [vocab.pad_token] * (max_clean - len(s)) for s in clean_encoded]
    
    return torch.tensor(noisy_padded, dtype=torch.long), torch.tensor(clean_padded, dtype=torch.long)

# 2. Main Training Loop
def train_model():
    # Automatically use Apple Silicon GPU (MPS), Nvidia GPU (CUDA), or fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on Hardware Accelerator: {device.type.upper()}")
    
    # Load Data
    dataset_path = os.path.join(os.path.dirname(__file__), "training_data.parquet")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Could not find {dataset_path}. Please run dataset_generator.py first!")
        return

    dataset = DNADataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn) # Reduced batch size for Mac VRAM
    
    # Initialize "World-Class" Model
    vocab = DNAVocabulary()
    model = DNADenoiserTransformer(vocab_size=vocab.vocab_size).to(device)
    
    # Loss & Optimizer
    # We explicitly tell the AI to ignore <PAD> tokens so it doesn't artificially lower the loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 3
    model.train()
    
    print("\n🔥 Starting Deep Learning Training Phase...")
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # "Teacher Forcing": We give the decoder the sequence up to the current nucleotide, 
            # and ask it to predict the *next* nucleotide.
            tgt_input = clean[:, :-1]
            tgt_expected = clean[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward Pass (Predict)
            logits = model(src=noisy, tgt=tgt_input)
            
            # Logits shape: (batch_size, seq_len, vocab_size)
            # Calculate loss (how wrong the AI was)
            loss = criterion(logits.reshape(-1, vocab.vocab_size), tgt_expected.reshape(-1))
            
            # Backward Pass (Learn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s\n")
        
    # Save the trained brain!
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, "dnadenoiser_sota.pth")
    torch.save(model.state_dict(), save_path)
    
    print(f"🎉 Training Complete! Neural Network weights permanently saved to: {save_path}")

if __name__ == "__main__":
    train_model()
