pub fn encode(data: &[u8]) -> String {
    let mut dna = String::new();
    let mut current_base = 0; // 0=A, 1=C, 2=G, 3=T
    
    // Process in 8-byte chunks
    let mut padded_data = data.to_vec();
    let padding_len = (8 - (data.len() % 8)) % 8;
    padded_data.resize(data.len() + padding_len, 0);
    
    for chunk in padded_data.chunks(8) {
        let mut val = u64::from_be_bytes(chunk.try_into().unwrap());
        let mut trits = [0u8; 41];
        
        for i in 0..41 {
            trits[40 - i] = (val % 3) as u8;
            val /= 3;
        }
        
        for trit in trits {
            current_base = (current_base + trit + 1) % 4;
            let nucleotide = match current_base {
                0 => 'A',
                1 => 'C',
                2 => 'G',
                _ => 'T',
            };
            dna.push(nucleotide);
        }
    }
    
    dna
}
