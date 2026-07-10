//! 5D Optical Memory Module
//! 
//! Implements femtosecond laser encoding for 5D optical storage in fused silica.
//! Features nano-grating orientation mapping and Jones calculus decoding.

pub mod femtosecond_voxel_mapper;
pub mod nanograting_orientation;
pub mod jones_calculus_decoder;

pub use femtosecond_voxel_mapper::FemtosecondVoxelMapper;
pub use nanograting_orientation::NanoGratingOrientation;
pub use jones_calculus_decoder::JonesCalculusDecoder;
