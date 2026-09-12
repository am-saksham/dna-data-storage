import numpy as np
import random

class BiologicalNoiseSimulator:
    def __init__(self, insertion_rate=0.02, deletion_rate=0.02, substitution_rate=0.01):
        """
        Initializes the noise simulator with specific error rates.
        Rates are probabilities per base.
        """
        self.p_ins = insertion_rate
        self.p_del = deletion_rate
        self.p_sub = substitution_rate
        self.bases = ['A', 'C', 'G', 'T']
        
    def add_noise(self, clean_dna: str) -> str:
        """
        Takes a clean DNA string and injects Insertions, Deletions, and Substitutions.
        """
        noisy_dna = []
        for base in clean_dna:
            # 1. Check for Deletion
            if random.random() < self.p_del:
                continue # Skip this base (deletion)
            
            # 2. Check for Substitution
            if random.random() < self.p_sub:
                mutated_base = random.choice([b for b in self.bases if b != base])
                noisy_dna.append(mutated_base)
            else:
                noisy_dna.append(base)
                
            # 3. Check for Insertion (after the current base)
            # We use a while loop because theoretically multiple insertions could happen
            while random.random() < self.p_ins:
                inserted_base = random.choice(self.bases)
                noisy_dna.append(inserted_base)
                
        return "".join(noisy_dna)

if __name__ == "__main__":
    # Test the simulator
    sim = BiologicalNoiseSimulator(insertion_rate=0.05, deletion_rate=0.05, substitution_rate=0.05)
    clean = "ACGTACGTACGTACGT"
    noisy = sim.add_noise(clean)
    print(f"Clean: {clean} (Len: {len(clean)})")
    print(f"Noisy: {noisy} (Len: {len(noisy)})")
