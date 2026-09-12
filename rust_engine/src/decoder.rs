pub fn decode(dna: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut prev_base = 0; // 0=A, 1=C, 2=G, 3=T
    
    let chars: Vec<char> = dna.chars().collect();
    let num_chunks = chars.len() / 41;
    
    for chunk_idx in 0..num_chunks {
        let mut val: u64 = 0;
        
        for i in 0..41 {
            let c = chars[chunk_idx * 41 + i];
            let current_base = match c {
                'A' => 0,
                'C' => 1,
                'G' => 2,
                _ => 3,
            };
            
            // Calculate trit: current_base = (prev_base + trit + 1) % 4
            let mut trit = (current_base as i32 - prev_base as i32 - 1) % 4;
            if trit < 0 {
                trit += 4;
            }
            
            val = val * 3 + (trit as u64);
            prev_base = current_base;
        }
        
        bytes.extend_from_slice(&val.to_be_bytes());
    }
    
    bytes
}
