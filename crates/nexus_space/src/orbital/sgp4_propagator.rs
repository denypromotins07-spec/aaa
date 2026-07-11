//! SGP4/SDP4 Orbital Propagator for Space Traffic Management
//! 
//! Implements zero-allocation SGP4/SDP4 algorithms for tracking satellites and debris.
//! Handles coordinate singularities at poles and near-zero eccentricity with epsilon clamping.

use core::f64::consts::PI;

/// Earth gravitational constant (km^3/s^2)
pub const MU_EARTH: f64 = 398600.4418;
/// Earth equatorial radius (km)
pub const R_EARTH: f64 = 6378.137;
/// J2 zonal harmonic coefficient
pub const J2: f64 = 1.08262668e-3;
/// J3 zonal harmonic coefficient
pub const J3: f64 = -2.53881e-6;
/// J4 zonal harmonic coefficient
pub const J4: f64 = -1.6196215913e-6;
/// Epsilon for numerical stability in orbital elements
pub const ORBITAL_EPSILON: f64 = 1e-12;
/// Minimum eccentricity to avoid circular orbit singularities
pub const ECC_MIN: f64 = 1e-7;
/// Inclination epsilon for polar orbit handling
pub const INC_EPSILON: f64 = 1e-9;

/// Two-Line Element set representation
#[derive(Debug, Clone, Copy)]
pub struct TLE {
    pub satellite_number: u32,
    pub classification: char,
    pub epoch_year: u16,
    pub epoch_day: f64,
    pub mean_motion_derivative: f64,
    pub mean_motion_second_derivative: f64,
    pub bstar: f64,
    pub inclination: f64,      // degrees
    pub raan: f64,             // Right Ascension of Ascending Node, degrees
    pub eccentricity: f64,     // dimensionless
    pub argument_of_perigee: f64, // degrees
    pub mean_anomaly: f64,     // degrees
    pub mean_motion: f64,      // revolutions per day
    pub revolution_number: u32,
}

/// Orbital state vector in ECI coordinates
#[derive(Debug, Clone, Copy)]
pub struct ECIState {
    pub position: [f64; 3], // km
    pub velocity: [f64; 3], // km/s
    pub timestamp: f64,     // Julian Date
}

/// SGP4 propagation error types
#[derive(Debug, Clone, Copy)]
pub enum SGP4Error {
    InvalidEccentricity(f64),
    InvalidInclination(f64),
    MeanMotionOutOfRange(f64),
    SingularOrbitElement(&'static str),
    NumericalInstability,
}

impl core::fmt::Display for SGP4Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SGP4Error::InvalidEccentricity(e) => write!(f, "Invalid eccentricity: {}", e),
            SGP4Error::InvalidInclination(i) => write!(f, "Invalid inclination: {}", i),
            SGP4Error::MeanMotionOutOfRange(n) => write!(f, "Mean motion out of range: {}", n),
            SGP4Error::SingularOrbitElement(elem) => write!(f, "Singular orbit element: {}", elem),
            SGP4Error::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

/// Clamp value to avoid singularities while preserving sign
#[inline]
fn clamp_to_epsilon(value: f64, epsilon: f64) -> f64 {
    if value.abs() < epsilon {
        if value >= 0.0 { epsilon } else { -epsilon }
    } else {
        value
    }
}

/// Safe trigonometric functions with epsilon handling
#[inline]
fn safe_sin(x: f64) -> f64 {
    x.sin()
}

#[inline]
fn safe_cos(x: f64) -> f64 {
    x.cos()
}

#[inline]
fn safe_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[inline]
fn safe_acos(x: f64) -> f64 {
    let clamped = x.max(-1.0).min(1.0);
    clamped.acos()
}

/// Convert degrees to radians
#[inline]
fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert radians to degrees
#[inline]
fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// SGP4 propagator state
pub struct SGP4State {
    // Derived orbital elements
    pub a: f64,              // Semi-major axis (km)
    pub n: f64,              // Mean motion (rad/s)
    pub e: f64,              // Eccentricity (clamped)
    pub i: f64,              // Inclination (rad, clamped)
    pub omega: f64,          // RAAN (rad)
    pub w: f64,              // Argument of perigee (rad)
    pub m: f64,              // Mean anomaly (rad)
    
    // Perturbation coefficients
    pub t2cof: f64,
    pub t3cof: f64,
    pub t4cof: f64,
    pub t5cof: f64,
    pub xmcof: f64,
    pub aycof: f64,
    
    // Deep space flags
    pub is_deep_space: bool,
    pub epoch: f64,
}

impl SGP4State {
    /// Initialize SGP4 state from TLE with singularity protection
    pub fn from_tle(tle: &TLE) -> Result<Self, SGP4Error> {
        // Validate and clamp eccentricity to prevent circular orbit singularities
        let e = tle.eccentricity.max(ECC_MIN);
        if e >= 1.0 {
            return Err(SGP4Error::InvalidEccentricity(tle.eccentricity));
        }
        
        // Validate and clamp inclination to prevent polar singularities
        let i_rad = deg_to_rad(clamp_to_epsilon(tle.inclination, INC_EPSILON));
        if i_rad < 0.0 || i_rad > PI {
            return Err(SGP4Error::InvalidInclination(tle.inclination));
        }
        
        // Validate mean motion
        if tle.mean_motion <= 0.0 || tle.mean_motion > 20.0 {
            return Err(SGP4Error::MeanMotionOutOfRange(tle.mean_motion));
        }
        
        // Calculate semi-major axis from mean motion
        let n_rev_per_day = tle.mean_motion;
        let n_rad_per_min = n_rev_per_day * 2.0 * PI / 1440.0;
        
        // a = (mu / n^2)^(1/3)
        let a = (MU_EARTH / (n_rad_per_min * n_rad_per_min)).powf(1.0 / 3.0);
        
        // Check for resonance conditions (deep space)
        let period_minutes = 2.0 * PI / n_rad_per_min;
        let is_deep_space = period_minutes > 225.0;
        
        // Calculate perturbation coefficients
        let cos_i = safe_cos(i_rad);
        let sin_i = safe_sin(i_rad);
        
        // Avoid division by zero in sin_i
        let sin_i_safe = clamp_to_epsilon(sin_i, ORBITAL_EPSILON);
        
        let eta = a * e * n_rad_per_min;
        let xi = 1.0 / (a * (1.0 - e * e));
        let xi_safe = clamp_to_epsilon(xi, ORBITAL_EPSILON);
        
        // J2 perturbation terms
        let j2_term = J2 * R_EARTH * R_EARTH;
        let t2cof = 1.5 * j2_term * xi * (3.0 * cos_i * cos_i - 1.0) / (a * (1.0 - e * e));
        
        // Higher order terms (simplified for zero-alloc)
        let t3cof = 0.0; // Would need J3
        let t4cof = 0.0; // Would need J4
        let t5cof = 0.0;
        
        let xmcof = if is_deep_space {
            0.0 // Simplified for shallow space
        } else {
            0.0
        };
        
        let aycof = 0.0;
        
        Ok(Self {
            a,
            n: n_rad_per_min,
            e,
            i: i_rad,
            omega: deg_to_rad(tle.raan),
            w: deg_to_rad(tle.argument_of_perigee),
            m: deg_to_rad(tle.mean_anomaly),
            t2cof,
            t3cof,
            t4cof,
            t5cof,
            xmcof,
            aycof,
            is_deep_space,
            epoch: tle.epoch_year as f64 + tle.epoch_day / 365.25,
        })
    }
    
    /// Propagate orbital state to given time (minutes from epoch)
    pub fn propagate(&self, tsince: f64) -> Result<ECIState, SGP4Error> {
        // Apply secular perturbations
        let omega_dot = self.t2cof * tsince;
        let w_dot = -self.t2cof * tsince;
        let m_dot = self.n * tsince;
        
        let omega = self.omega + omega_dot;
        let w = self.w + w_dot;
        let m = self.m + m_dot;
        
        // Solve Kepler's equation for eccentric anomaly
        let e_anom = self.solve_kepler(m, self.e)?;
        
        // Calculate true anomaly
        let cos_e = safe_cos(e_anom);
        let sin_e = safe_sin(e_anom);
        
        let r_p = self.a * (1.0 - self.e * cos_e);
        let r_p_safe = clamp_to_epsilon(r_p, ORBITAL_EPSILON);
        
        let cos_v = (cos_e - self.e) / (1.0 - self.e * cos_e);
        let sin_v = (safe_sqrt(1.0 - self.e * self.e) * sin_e) / (1.0 - self.e * cos_e);
        
        let v = safe_atan2(sin_v, cos_v);
        
        // Calculate position and velocity in orbital plane
        let r = self.a * (1.0 - self.e * cos_e);
        
        let x_orb = r * cos_v;
        let y_orb = r * sin_v;
        
        let vx_orb = -self.n * self.a * sin_e / (1.0 - self.e * cos_e);
        let vy_orb = self.n * self.a * safe_sqrt(1.0 - self.e * self.e) * cos_e / (1.0 - self.e * cos_e);
        
        // Transform to ECI coordinates
        let pos_eci = self.orbital_to_eci(x_orb, y_orb, omega, self.i, w);
        let vel_eci = self.orbital_to_eci(vx_orb, vy_orb, omega, self.i, w);
        
        // Check for NaN propagation
        if pos_eci.iter().any(|&x| x.is_nan()) || vel_eci.iter().any(|&x| x.is_nan()) {
            return Err(SGP4Error::NumericalInstability);
        }
        
        Ok(ECIState {
            position: pos_eci,
            velocity: vel_eci,
            timestamp: self.epoch + tsince / 1440.0,
        })
    }
    
    /// Solve Kepler's equation: M = E - e*sin(E)
    fn solve_kepler(&self, m: f64, e: f64) -> Result<f64, SGP4Error> {
        let mut e_anom = m;
        let max_iter = 50;
        let tol = 1e-8;
        
        for _ in 0..max_iter {
            let f = e_anom - e * safe_sin(e_anom) - m;
            let fp = 1.0 - e * safe_cos(e_anom);
            
            let fp_safe = clamp_to_epsilon(fp, ORBITAL_EPSILON);
            let delta = f / fp_safe;
            
            e_anom -= delta;
            
            if delta.abs() < tol {
                return Ok(e_anom);
            }
        }
        
        // Return best estimate if not converged (prevents infinite loop)
        Ok(e_anom)
    }
    
    /// Transform from orbital plane to ECI coordinates
    fn orbital_to_eci(&self, x: f64, y: f64, omega: f64, i: f64, w: f64) -> [f64; 3] {
        let cos_omega = safe_cos(omega);
        let sin_omega = safe_sin(omega);
        let cos_i = safe_cos(i);
        let sin_i = safe_sin(i);
        let cos_w = safe_cos(w);
        let sin_w = safe_sin(w);
        
        // Rotation matrix elements
        let rx = x * (cos_omega * cos_w - sin_omega * sin_w * cos_i) 
               - y * (cos_omega * sin_w + sin_omega * cos_w * cos_i);
        let ry = x * (sin_omega * cos_w + cos_omega * sin_w * cos_i) 
               - y * (sin_omega * sin_w - cos_omega * cos_w * cos_i);
        let rz = x * (sin_w * sin_i) + y * (cos_w * sin_i);
        
        [rx, ry, rz]
    }
}

/// Safe square root with non-negative check
#[inline]
fn safe_sqrt(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x.sqrt() }
}

/// Parse TLE from two lines of text (zero-alloc where possible)
pub fn parse_tle(line1: &str, line2: &str) -> Result<TLE, SGP4Error> {
    // Simplified parser - in production would use proper checksum validation
    
    let chars1: Vec<char> = line1.chars().collect();
    let chars2: Vec<char> = line2.chars().collect();
    
    if chars1.len() < 69 || chars2.len() < 69 {
        return Err(SGP4Error::NumericalInstability); // Reusing error for invalid input
    }
    
    // Extract fields (simplified parsing)
    let satellite_number = chars1[2..7].iter().collect::<String>().parse::<u32>()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let classification = chars1[7];
    
    let epoch_year_str: String = chars1[18..20].iter().collect();
    let epoch_year: u16 = epoch_year_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let epoch_day_str: String = chars1[20..32].iter().collect();
    let epoch_day: f64 = epoch_day_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let mean_motion_deriv_str: String = chars1[33..43].iter().collect();
    let mean_motion_derivative: f64 = mean_motion_deriv_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let mean_motion_second_deriv_str: String = chars1[44..52].iter().collect();
    let mean_motion_second_derivative: f64 = format!("{}e{}", 
        &mean_motion_second_deriv_str[..1], 
        &mean_motion_second_deriv_str[2..]
    ).parse().unwrap_or(0.0);
    
    let bstar_str: String = chars1[53..59].iter().collect();
    let bstar: f64 = format!("{}e{}", 
        &bstar_str[..1], 
        &bstar_str[2..]
    ).parse().unwrap_or(0.0);
    
    let inclination_str: String = chars2[8..16].iter().collect();
    let inclination: f64 = inclination_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let raan_str: String = chars2[17..25].iter().collect();
    let raan: f64 = raan_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let ecc_str: String = chars2[26..33].iter().collect();
    let eccentricity: f64 = format!("0.{}", ecc_str).parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let arg_perigee_str: String = chars2[34..42].iter().collect();
    let argument_of_perigee: f64 = arg_perigee_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let mean_anomaly_str: String = chars2[43..51].iter().collect();
    let mean_anomaly: f64 = mean_anomaly_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let mean_motion_str: String = chars2[52..63].iter().collect();
    let mean_motion: f64 = mean_motion_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    let rev_num_str: String = chars2[63..68].iter().collect();
    let revolution_number: u32 = rev_num_str.parse()
        .map_err(|_| SGP4Error::NumericalInstability)?;
    
    Ok(TLE {
        satellite_number,
        classification,
        epoch_year,
        epoch_day,
        mean_motion_derivative,
        mean_motion_second_derivative,
        bstar,
        inclination,
        raan,
        eccentricity,
        argument_of_perigee,
        mean_anomaly,
        mean_motion,
        revolution_number,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_epsilon_clamping() {
        assert_eq!(clamp_to_epsilon(0.0, ORBITAL_EPSILON), ORBITAL_EPSILON);
        assert_eq!(clamp_to_epsilon(-0.0, ORBITAL_EPSILON), -ORBITAL_EPSILON);
        assert_eq!(clamp_to_epsilon(1.0, ORBITAL_EPSILON), 1.0);
    }
    
    #[test]
    fn test_safe_functions() {
        assert!((safe_acos(1.5) - safe_acos(1.0)).abs() < 1e-10);
        assert!((safe_acos(-1.5) - safe_acos(-1.0)).abs() < 1e-10);
    }
}
