use std::io::Cursor;
use zstd::stream::{encode_all, decode_all};

pub fn compress(data: &[u8]) -> Result<Vec<u8>, String> {
    // 0 is the default compression level (usually 3)
    // For maximum density in DNA, we use a high level like 19 for maximum compression.
    let level = 19;
    encode_all(Cursor::new(data), level).map_err(|e| format!("Compression failed: {}", e))
}

pub fn decompress(compressed_data: &[u8]) -> Result<Vec<u8>, String> {
    decode_all(Cursor::new(compressed_data)).map_err(|e| format!("Decompression failed: {}", e))
}
