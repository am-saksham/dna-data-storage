import os
import sys
import random
import pandas as pd
from noise_simulator import BiologicalNoiseSimulator

# Ensure we can import the compiled rust module from the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rust_engine", "target", "wheels")) 
import dna_codec

def generate_random_bytes(length=32):
    return bytes(random.getrandbits(8) for _ in range(length))

def generate_dataset(num_samples=100_000, output_file="training_data.parquet"):
    print(f"Generating {num_samples} DNA sequences...")
    sim = BiologicalNoiseSimulator(insertion_rate=0.02, deletion_rate=0.02, substitution_rate=0.01)
    
    clean_sequences = []
    noisy_sequences = []
    
    for i in range(num_samples):
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i}/{num_samples} samples...")
            
        raw_bytes = generate_random_bytes(length=64) # 64 bytes per payload
        
        # 1. Rust encoding
        try:
            clean_dna = dna_codec.encode_with_rs(raw_bytes)
        except Exception as e:
            continue
            
        # 2. Add biological noise
        noisy_dna = sim.add_noise(clean_dna)
        
        clean_sequences.append(clean_dna)
        noisy_sequences.append(noisy_dna)
        
    print("Packing into DataFrame...")
    df = pd.DataFrame({
        "noisy_dna": noisy_sequences,
        "clean_dna": clean_sequences
    })
    
    print(f"Saving to {output_file} via PyArrow...")
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    print(f"Dataset saved successfully! Size on disk: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    generate_dataset(num_samples=100_000)
