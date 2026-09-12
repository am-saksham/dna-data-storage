import sys
import os

# Ensure we can import the compiled module
sys.path.append(os.path.join(os.path.dirname(__file__), "dna_codec", "target", "wheels")) 

import dna_codec
import random

def test():
    # A highly compressible string to prove zstd works
    original_data = b"Hello DNA Data Storage World! " * 50
    print(f"--- Original bytes (Length: {len(original_data)}) ---")
    
    # 1. Encode with just RS (No Compression)
    uncompressed_dna = dna_codec.encode_with_rs(original_data)
    print(f"\n--- Uncompressed Encoded DNA ({len(uncompressed_dna)} bases) ---")
    
    # 2. Encode with Full Pipeline (Compression -> RS -> DNA)
    compressed_dna = dna_codec.encode_full_pipeline(original_data)
    print(f"--- Compressed Encoded DNA ({len(compressed_dna)} bases) ---")
    print(f"-> ZSTD Compression achieved a {(len(uncompressed_dna) - len(compressed_dna)) / len(uncompressed_dna) * 100:.1f}% reduction in DNA synthesis cost!")
    
    print(f"\n--- DNA Constraints Verification ---")
    gc_content = dna_codec.get_gc_content(compressed_dna)
    max_homopolymer = dna_codec.get_max_homopolymer(compressed_dna)
    print(f"GC Content: {gc_content * 100:.2f}% (Target: ~50%)")
    print(f"Max Homopolymer Length: {max_homopolymer} (Target: 1)")
    
    if max_homopolymer > 1:
        print("ERROR: Homopolymers detected!")
    
    # 3. Simulate Biological Mutation on the Compressed DNA
    dna_list = list(compressed_dna)
    mutations = 4
    for _ in range(mutations):
        idx = random.randint(0, len(dna_list) - 1)
        current_base = dna_list[idx]
        new_base = random.choice([b for b in 'ACGT' if b != current_base])
        dna_list[idx] = new_base
        
    mutated_dna = "".join(dna_list)
    print(f"\n--- Mutated Compressed DNA ({mutations} biological substitutions injected) ---")
    
    # 4. Decode the mutated DNA through the full pipeline
    try:
        decoded_data = bytes(dna_codec.decode_full_pipeline(mutated_dna))
        
        # Verify
        if original_data == decoded_data:
            print("\nSUCCESS: Data Losslessly Recovered!")
            print("Pipeline: Biological Mutations Fixed (RS) -> Decompressed (ZSTD) -> Original Bytes")
        else:
            print("\nERROR: Data mismatch!")
    except Exception as e:
        print(f"\nDECODING FAILED: {e}")

if __name__ == "__main__":
    test()
