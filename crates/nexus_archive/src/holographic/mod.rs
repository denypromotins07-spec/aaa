//! Holographic Volumetric Storage Module
//! 
//! Implements 3D voxel grid encoding for holographic crystal storage.
//! Maps 2D financial data matrices into volumetric pages with Bragg grating simulation.

pub mod volumetric_page_encoder;
pub mod bragg_grating_simulator;
pub mod angular_multiplexing;

pub use volumetric_page_encoder::VolumetricPageEncoder;
pub use bragg_grating_simulator::BraggGratingSimulator;
pub use angular_multiplexing::HolographicMultiplexer;
