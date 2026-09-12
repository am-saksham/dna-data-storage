use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod encoder;
mod decoder;
mod constraints;
mod error_correction;
mod compression;

/// Encodes a byte array into a DNA string with Compression -> Reed-Solomon -> DNA.
#[pyfunction]
fn encode_full_pipeline(data: &[u8]) -> PyResult<String> {
    // 1. Compress
    let compressed_data = compression::compress(data).map_err(PyValueError::new_err)?;
    
    // 2. Error Correction Parity
    let rs_payload = error_correction::rs_encode(&compressed_data);
    
    // 3. DNA Encoding
    Ok(encoder::encode(&rs_payload))
}

/// Decodes a DNA string back into a byte array, fixing mutations and decompressing.
#[pyfunction]
fn decode_full_pipeline(dna: &str) -> PyResult<Vec<u8>> {
    // 1. DNA Decoding
    let rs_payload = decoder::decode(dna);
    
    // 2. Error Correction (Fixes biological mutations)
    let compressed_data = error_correction::rs_decode(&rs_payload).map_err(PyValueError::new_err)?;
    
    // 3. Decompress
    compression::decompress(&compressed_data).map_err(PyValueError::new_err)
}

/// Encodes a byte array into a DNA string with Reed-Solomon parity.
#[pyfunction]
fn encode_with_rs(data: &[u8]) -> PyResult<String> {
    let rs_payload = error_correction::rs_encode(data);
    Ok(encoder::encode(&rs_payload))
}

/// Decodes a DNA string back into a byte array, correcting any RS errors.
#[pyfunction]
fn decode_with_rs(dna: &str) -> PyResult<Vec<u8>> {
    let rs_payload = decoder::decode(dna);
    match error_correction::rs_decode(&rs_payload) {
        Ok(data) => Ok(data),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Encodes a byte array into a DNA string.
#[pyfunction]
fn encode_to_dna(data: &[u8]) -> PyResult<String> {
    Ok(encoder::encode(data))
}

/// Decodes a DNA string back into a byte array.
#[pyfunction]
fn decode_from_dna(dna: &str) -> PyResult<Vec<u8>> {
    Ok(decoder::decode(dna))
}

/// Calculates GC content
#[pyfunction]
fn get_gc_content(dna: &str) -> PyResult<f64> {
    Ok(constraints::calculate_gc_content(dna))
}

/// Gets max homopolymer length
#[pyfunction]
fn get_max_homopolymer(dna: &str) -> PyResult<usize> {
    Ok(constraints::max_homopolymer_length(dna))
}

/// A Python module implemented in Rust.
#[pymodule]
fn dna_codec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_to_dna, m)?)?;
    m.add_function(wrap_pyfunction!(decode_from_dna, m)?)?;
    m.add_function(wrap_pyfunction!(encode_with_rs, m)?)?;
    m.add_function(wrap_pyfunction!(decode_with_rs, m)?)?;
    m.add_function(wrap_pyfunction!(encode_full_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(decode_full_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(get_gc_content, m)?)?;
    m.add_function(wrap_pyfunction!(get_max_homopolymer, m)?)?;
    Ok(())
}
