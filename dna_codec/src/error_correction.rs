use reed_solomon::Encoder;
use reed_solomon::Decoder;

const ECC_LEN: usize = 128;
const DATA_CHUNK_SIZE: usize = 255 - ECC_LEN;

pub fn rs_encode(data: &[u8]) -> Vec<u8> {
    let encoder = Encoder::new(ECC_LEN);
    
    // Prefix with 4-byte length
    let mut payload = Vec::with_capacity(data.len() + 4);
    payload.extend_from_slice(&(data.len() as u32).to_be_bytes());
    payload.extend_from_slice(data);
    
    let mut encoded = Vec::new();
    for chunk in payload.chunks(DATA_CHUNK_SIZE) {
        // Pad the chunk to DATA_CHUNK_SIZE if it's the last one
        let mut padded_chunk = chunk.to_vec();
        padded_chunk.resize(DATA_CHUNK_SIZE, 0);
        let block = encoder.encode(&padded_chunk);
        encoded.extend_from_slice(&block);
    }
    encoded
}

pub fn rs_decode(encoded_payload: &[u8]) -> Result<Vec<u8>, String> {
    let decoder = Decoder::new(ECC_LEN);
    let mut decoded_payload = Vec::new();
    
    for block in encoded_payload.chunks(255) {
        if block.len() != 255 {
            // These are trailing padding bytes added by the 8-byte chunking in the DNA encoder layer.
            // We can safely ignore them.
            break;
        }
        match decoder.correct(block, None) {
            Ok(decoded) => {
                decoded_payload.extend_from_slice(&decoded.data()[..DATA_CHUNK_SIZE]);
            },
            Err(_) => return Err("Too many errors to correct in a block".to_string()),
        }
    }
    
    if decoded_payload.len() < 4 {
        return Err("Payload too short".to_string());
    }
    
    // Read the original length
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&decoded_payload[0..4]);
    let original_len = u32::from_be_bytes(len_bytes) as usize;
    
    if original_len + 4 > decoded_payload.len() {
        return Err("Decoded length mismatch".to_string());
    }
    
    Ok(decoded_payload[4..4 + original_len].to_vec())
}
