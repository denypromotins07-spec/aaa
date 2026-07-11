//! Chapter 4: Interstellar Laser Propagation & Light-Cone Archiving

pub mod shannon_hartley_laser;
pub mod doppler_pre_compensation;
pub mod plasma_dispersion_model;

pub use shannon_hartley_laser::ShannonHartleyEncoder;
pub use doppler_pre_compensation::DopplerCompensator;
pub use plasma_dispersion_model::PlasmaDispersionModel;
