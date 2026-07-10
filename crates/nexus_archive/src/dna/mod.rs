//! DNA Data Storage Module
//! 
//! Implements synthetic DNA encoding with biological error correction.
//! Features homopolymer avoidance, GC-content balancing, and Reed-Solomon over GF(4).

pub mod nucleotide_base4_encoder;
pub mod homopolymer_avoidance;
pub mod reed_solomon_gf4;

pub use nucleotide_base4_encoder::NucleotideEncoder;
pub use homopolymer_avoidance::HomopolymerAvoidanceEncoder;
pub use reed_solomon_gf4::ReedSolomonGF4;
