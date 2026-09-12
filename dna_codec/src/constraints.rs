pub fn calculate_gc_content(dna: &str) -> f64 {
    if dna.is_empty() {
        return 0.0;
    }
    let gc_count = dna.chars().filter(|&c| c == 'G' || c == 'C').count();
    (gc_count as f64) / (dna.len() as f64)
}

pub fn max_homopolymer_length(dna: &str) -> usize {
    if dna.is_empty() {
        return 0;
    }
    let mut max_len = 1;
    let mut current_len = 1;
    let mut prev_char = dna.chars().next().unwrap();

    for c in dna.chars().skip(1) {
        if c == prev_char {
            current_len += 1;
            if current_len > max_len {
                max_len = current_len;
            }
        } else {
            current_len = 1;
            prev_char = c;
        }
    }
    max_len
}
