import os
import sys
import time
import math

# Ensure we can import the compiled rust module
sys.path.append(os.path.join(os.path.dirname(__file__), "dna_codec", "target", "wheels")) 
import dna_codec

# ANSI color codes for that beautiful terminal aesthetic
COLORS = {
    'A': '\033[92m', # Green
    'C': '\033[96m', # Cyan
    'G': '\033[93m', # Yellow
    'T': '\033[91m', # Red
    'RESET': '\033[0m',
    'BRIDGE': '\033[90m' # Dark Gray for the bridges
}

def get_complement(base):
    return {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}.get(base, 'N')

def animate_helix(dna_string, speed=0.03):
    """Animates a scrolling 3D double helix matching the exact generated DNA."""
    print("\n" + "="*50)
    print("        🧬 SYNTHESIZING PHYSICAL DNA 🧬")
    print("="*50 + "\n")
    
    # We will only animate the first 200 bases so the user doesn't wait forever
    # But it will give the perfect effect
    display_length = min(len(dna_string), 200)
    
    for i in range(display_length):
        base = dna_string[i]
        comp = get_complement(base)
        
        # Apply colors
        c_base = f"{COLORS[base]}{base}{COLORS['RESET']}"
        c_comp = f"{COLORS[comp]}{comp}{COLORS['RESET']}"
        
        # Calculate sine wave to simulate 3D rotation
        t = i * 0.3 # Rotation speed
        amplitude = 12
        center = 25
        
        x1 = int(center + amplitude * math.sin(t))
        x2 = int(center + amplitude * math.sin(t + math.pi))
        
        # Determine which strand is on the left
        if x1 > x2:
            x1, x2 = x2, x1
            left_char, right_char = c_comp, c_base
        else:
            left_char, right_char = c_base, c_comp
            
        gap = x2 - x1 - 1
        
        if gap < 0:
            # The strands cross over each other
            line = " " * x1 + left_char
        elif gap == 0:
            # Touching
            line = " " * x1 + left_char + right_char
        else:
            # Draw the hydrogen bonds bridging the two strands
            bridge = f"{COLORS['BRIDGE']}" + "-" * gap + f"{COLORS['RESET']}"
            line = " " * x1 + left_char + bridge + right_char
            
        print(line)
        sys.stdout.flush() # Force terminal to print immediately
        time.sleep(speed) # The "motion" delay
        
    if len(dna_string) > 200:
        print(f"\n... and {len(dna_string) - 200:,} more base pairs ...\n")
        
    print("="*50 + "\n")

def encode_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    print(f"\n🧬 Loading file: {filepath}")
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
        
    print(f"💾 Original Size: {len(file_bytes):,} bytes")
    print("⚙️  Running Rust Codec Engine (ZSTD -> RS -> Base-3)...")
    
    start_time = time.time()
    dna_string = dna_codec.encode_full_pipeline(file_bytes)
    elapsed = time.time() - start_time
    
    print(f"✅ Encoding Complete in {elapsed:.5f} seconds!")
    print(f"🧬 Total DNA Sequence Length: {len(dna_string):,} nucleotides")
    
    gc_content = dna_codec.get_gc_content(dna_string)
    max_homopolymer = dna_codec.get_max_homopolymer(dna_string)
    print(f"📊 Constraints: GC Content = {gc_content*100:.2f}% | Max Homopolymer = {max_homopolymer}")
    
    # Run the motion animation!
    animate_helix(dna_string)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        encode_file(sys.argv[1])
    else:
        demo_text = "This is a demonstration of the DNA Data Storage Codec Engine. " * 50
        demo_path = "demo.txt"
        with open(demo_path, "w") as f:
            f.write(demo_text)
        encode_file(demo_path)
