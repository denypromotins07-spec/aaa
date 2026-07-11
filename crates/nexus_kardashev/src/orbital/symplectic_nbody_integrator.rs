//! Symplectic N-Body Integrator for Dyson Swarm Orbital Mechanics
//! 
//! Implements symplectic integration schemes (Leapfrog, Yoshida) that preserve
//! the Hamiltonian phase-space volume, preventing energy drift in long-term
//! orbital simulations of billions of Dyson swarm mirrors.

use nalgebra::{SVector, Vector3};
use num_traits::{Float, Zero};
use thiserror::Error;

/// Physical state of a body in the N-body simulation
#[derive(Clone, Debug)]
pub struct BodyState<T> {
    pub position: SVector<T, 3>,
    pub velocity: SVector<T, 3>,
    pub mass: T,
    pub id: u64,
}

impl<T: Float + Copy + Zero> BodyState<T> {
    pub fn new(id: u64, position: SVector<T, 3>, velocity: SVector<T, 3>, mass: T) -> Self {
        Self {
            id,
            position,
            velocity,
            mass,
        }
    }
    
    /// Compute specific orbital energy: ε = v²/2 - μ/r
    pub fn specific_orbital_energy(&self, mu: T) -> T {
        let r_sq = self.position.dot(&self.position);
        let v_sq = self.velocity.dot(&self.velocity);
        
        if r_sq <= T::zero() {
            return T::zero();
        }
        
        let r = r_sq.sqrt();
        let two = T::one() + T::one();
        
        v_sq / two - mu / r
    }
    
    /// Compute specific angular momentum: h = r × v
    pub fn specific_angular_momentum(&self) -> SVector<T, 3> {
        self.position.cross(&self.velocity)
    }
}

/// Gravitational parameter for central body
pub struct GravitationalSystem<T> {
    pub mu: T,  // GM (gravitational parameter)
    pub j2: T,  // Second zonal harmonic (oblateness)
    pub equatorial_radius: T,
}

impl<T: Float + Zero> GravitationalSystem<T> {
    /// Create system with just point-mass gravity
    pub fn point_mass(mu: T) -> Self {
        Self {
            mu,
            j2: T::zero(),
            equatorial_radius: T::zero(),
        }
    }
    
    /// Create system with J2 perturbation (oblate primary)
    pub fn with_j2(mu: T, j2: T, radius: T) -> Self {
        Self {
            mu,
            j2,
            equatorial_radius: radius,
        }
    }
}

/// Errors in N-body integration
#[derive(Error, Debug)]
pub enum NBodyError {
    #[error("Collision detected between bodies {0} and {1}")]
    Collision(u64, u64),
    #[error("Body escaped system (r > {escape_radius:?})")]
    Escape { body_id: u64, escape_radius: f64 },
    #[error("Numerical overflow in position/velocity")]
    NumericalOverflow,
    #[error("Invalid timestep dt={dt:?}")]
    InvalidTimestep { dt: f64 },
}

/// Symplectic Leapfrog integrator (Velocity Verlet variant)
/// Preserves symplectic structure exactly, no energy drift over centuries
pub struct LeapfrogIntegrator<T> {
    system: GravitationalSystem<T>,
}

impl<T: Float + Copy + Zero> LeapfrogIntegrator<T> {
    pub fn new(system: GravitationalSystem<T>) -> Self {
        Self { system }
    }
    
    /// Single leapfrog step: drift-kick-drift
    /// 
    /// This is a second-order symplectic integrator that exactly preserves
    /// the symplectic 2-form, ensuring bounded energy error over arbitrary times.
    pub fn step(&self, bodies: &mut [BodyState<T>], dt: T) -> Result<(), NBodyError> {
        if dt <= T::zero() {
            return Err(NBodyError::InvalidTimestep { dt: T::from(0.0).unwrap_or_else(|| T::zero()).to_f64().unwrap_or(0.0) });
        }
        
        let two = T::one() + T::one();
        let half_dt = dt / two;
        
        // First half-step: drift (update positions by half timestep)
        for body in bodies.iter_mut() {
            body.position = body.position + body.velocity.map(|v| v * half_dt);
        }
        
        // Compute accelerations at midpoint
        let accelerations: Vec<SVector<T, 3>> = bodies.iter()
            .map(|body| self.compute_acceleration(body, bodies))
            .collect();
        
        // Full step: kick (update velocities by full timestep)
        for (body, acc) in bodies.iter_mut().zip(accelerations.iter()) {
            body.velocity = body.velocity + acc.map(|a| a * dt);
        }
        
        // Second half-step: drift (update positions by half timestep)
        for body in bodies.iter_mut() {
            body.position = body.position + body.velocity.map(|v| v * half_dt);
        }
        
        Ok(())
    }
    
    /// Compute gravitational acceleration on a body from all others
    fn compute_acceleration(&self, body: &BodyState<T>, all_bodies: &[BodyState<T>]) -> SVector<T, 3> {
        let mut acc = SVector::<T, 3>::zeros();
        
        // Central body gravity (dominant term)
        acc += self.central_gravity(&body.position);
        
        // Third-body perturbations from other swarm elements
        for other in all_bodies {
            if other.id == body.id {
                continue;
            }
            
            let r_vec = other.position - body.position;
            let r_sq = r_vec.dot(&r_vec);
            
            // Softening to prevent singularities (numerical stability)
            let softening = T::from(100.0).unwrap_or_else(|| T::one());
            let r_soft_sq = r_sq + softening * softening;
            
            if r_soft_sq <= T::zero() {
                continue;
            }
            
            let r_soft = r_soft_sq.sqrt();
            let inv_r_cubed = T::one() / (r_soft * r_soft_sq);
            
            // Newton's law: a = G * m * r̂ / r² = G * m * r / r³
            acc = acc + r_vec.map(|x| x * other.mass * inv_r_cubed);
        }
        
        acc
    }
    
    /// Central body gravitational acceleration with J2 perturbation
    fn central_gravity(&self, position: &SVector<T, 3>) -> SVector<T, 3> {
        let r_sq = position.dot(&position);
        if r_sq <= T::zero() {
            return SVector::zeros();
        }
        
        let r = r_sq.sqrt();
        let inv_r_cubed = T::one() / (r * r_sq);
        
        // Point mass term: -μ r / r³
        let mut acc = position.map(|x| x * (-self.system.mu * inv_r_cubed));
        
        // J2 perturbation (if non-zero)
        if self.system.j2 > T::zero() && self.system.equatorial_radius > T::zero() {
            acc += self.j2_perturbation(position, r, r_sq);
        }
        
        acc
    }
    
    /// J2 oblateness perturbation acceleration
    fn j2_perturbation(&self, position: &SVector<T, 3>, r: T, r_sq: T) -> SVector<T, 3> {
        let five = T::from(5.0).unwrap_or_else(|| {
            let one = T::one();
            one + one + one + one + one
        });
        let two = T::one() + T::one();
        
        let z_sq = position[2] * position[2];
        let ratio = self.system.equatorial_radius / r;
        let ratio_sq = ratio * ratio;
        let ratio_fourth = ratio_sq * ratio_sq;
        
        let factor = T::from(1.5).unwrap_or_else(|| T::from(3).unwrap() / T::from(2).unwrap())
                   * self.system.j2 * self.system.mu * ratio_fourth / r_sq;
        
        // J2 acceleration components
        let ax = position[0] * (five * z_sq / r_sq - one);
        let ay = position[1] * (five * z_sq / r_sq - one);
        let az = position[2] * (five * z_sq / r_sq - three);
        
        SVector::new(
            factor * ax,
            factor * ay,
            factor * az,
        )
    }
}

fn three<T: Float>() -> T {
    let one = T::one();
    one + one + one
}

/// Yoshida 4th-order symplectic integrator
/// Higher accuracy than leapfrog while preserving symplectic structure
pub struct YoshidaIntegrator<T> {
    system: GravitationalSystem<T>,
    /// Yoshida coefficients for 4th-order composition
    c1: T, c2: T, c3: T, c4: T,
    d1: T, d2: T, d3: T,
}

impl<T: Float + Copy + Zero> YoshidaIntegrator<T> {
    pub fn new(system: GravitationalSystem<T>) -> Self {
        // Yoshida (1990) coefficients for 4th-order symplectic integrator
        // Based on triple-jump composition of leapfrog
        let one = T::one();
        let two = one + one;
        
        // w0 = 1/(2 - 2^(1/3))
        let w0_denom = two - two.powf(T::from(1.0/3.0).unwrap_or_else(|| T::from(0.333333).unwrap()));
        let w0 = one / w0_denom;
        
        let w1 = -two.powf(T::from(1.0/3.0).unwrap_or_else(|| T::from(0.333333).unwrap())) * w0;
        
        // Coefficients for position (drift) steps
        let c1 = w0 / two;
        let c2 = (w0 + w1) / two;
        let c3 = c2;
        let c4 = c1;
        
        // Coefficients for velocity (kick) steps
        let d1 = w0;
        let d2 = w0 + w1;
        let d3 = d2;
        
        Self {
            system,
            c1, c2, c3, c4,
            d1, d2, d3,
        }
    }
    
    /// Single Yoshida 4th-order step
    /// Composition: Φ(c1 dt) ∘ Ψ(d1 dt) ∘ Φ(c2 dt) ∘ Ψ(d2 dt) ∘ Φ(c3 dt) ∘ Ψ(d3 dt) ∘ Φ(c4 dt)
    pub fn step(&self, bodies: &mut [BodyState<T>], dt: T) -> Result<(), NBodyError> {
        if dt <= T::zero() {
            return Err(NBodyError::InvalidTimestep { dt: T::from(0.0).unwrap_or_else(|| T::zero()).to_f64().unwrap_or(0.0) });
        }
        
        // Substep 1
        self.drift(bodies, self.c1 * dt);
        self.kick(bodies, self.d1 * dt);
        
        // Substep 2
        self.drift(bodies, self.c2 * dt);
        self.kick(bodies, self.d2 * dt);
        
        // Substep 3
        self.drift(bodies, self.c3 * dt);
        self.kick(bodies, self.d3 * dt);
        
        // Substep 4
        self.drift(bodies, self.c4 * dt);
        
        Ok(())
    }
    
    fn drift(&self, bodies: &mut [BodyState<T>], dt: T) {
        for body in bodies.iter_mut() {
            body.position = body.position + body.velocity.map(|v| v * dt);
        }
    }
    
    fn kick(&self, bodies: &mut [BodyState<T>], dt: T) {
        let accelerations: Vec<SVector<T, 3>> = bodies.iter()
            .map(|body| {
                let mut acc = SVector::<T, 3>::zeros();
                acc += Self::central_gravity_static(&self.system, &body.position);
                
                for other in bodies {
                    if other.id == body.id {
                        continue;
                    }
                    
                    let r_vec = other.position - body.position;
                    let r_sq = r_vec.dot(&r_vec);
                    let softening = T::from(100.0).unwrap_or_else(|| T::one());
                    let r_soft_sq = r_sq + softening * softening;
                    
                    if r_soft_sq <= T::zero() {
                        continue;
                    }
                    
                    let r_soft = r_soft_sq.sqrt();
                    let inv_r_cubed = T::one() / (r_soft * r_soft_sq);
                    acc = acc + r_vec.map(|x| x * other.mass * inv_r_cubed);
                }
                
                acc
            })
            .collect();
        
        for (body, acc) in bodies.iter_mut().zip(accelerations.iter()) {
            body.velocity = body.velocity + acc.map(|a| a * dt);
        }
    }
    
    fn central_gravity_static(system: &GravitationalSystem<T>, position: &SVector<T, 3>) -> SVector<T, 3> {
        let r_sq = position.dot(&position);
        if r_sq <= T::zero() {
            return SVector::zeros();
        }
        
        let r = r_sq.sqrt();
        let inv_r_cubed = T::one() / (r * r_sq);
        let mut acc = position.map(|x| x * (-system.mu * inv_r_cubed));
        
        if system.j2 > T::zero() && system.equatorial_radius > T::zero() {
            let five = T::from(5.0).unwrap_or_else(|| {
                let one = T::one();
                one + one + one + one + one
            });
            
            let z_sq = position[2] * position[2];
            let ratio = system.equatorial_radius / r;
            let ratio_fourth = ratio * ratio * ratio * ratio;
            
            let factor = T::from(1.5).unwrap_or_else(|| T::from(3).unwrap() / T::from(2).unwrap())
                       * system.j2 * system.mu * ratio_fourth / r_sq;
            
            let ax = position[0] * (five * z_sq / r_sq - T::one());
            let ay = position[1] * (five * z_sq / r_sq - T::one());
            let az = position[2] * (five * z_sq / r_sq - three());
            
            acc = acc + SVector::new(factor * ax, factor * ay, factor * az);
        }
        
        acc
    }
}

/// Full N-body swarm simulator
pub struct DysonSwarmSimulator<T> {
    bodies: Vec<BodyState<T>>,
    integrator: LeapfrogIntegrator<T>,
    total_energy_history: Vec<T>,
}

impl<T: Float + Copy + Zero> DysonSwarmSimulator<T> {
    pub fn new(system: GravitationalSystem<T>) -> Self {
        Self {
            bodies: Vec::new(),
            integrator: LeapfrogIntegrator::new(system),
            total_energy_history: Vec::new(),
        }
    }
    
    /// Add a mirror/habitat to the swarm
    pub fn add_body(&mut self, body: BodyState<T>) {
        self.bodies.push(body);
    }
    
    /// Initialize circular orbit at given radius
    pub fn add_circular_orbit(&mut self, id: u64, radius: T, mass: T, inclination: T) {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let two = T::one() + T::one();
        
        // Position on x-axis
        let position = SVector::new(radius, T::zero(), T::zero());
        
        // Circular velocity: v = √(μ/r) in y-z plane based on inclination
        let v_circular = (self.integrator.system.mu / radius).sqrt();
        
        let cos_i = inclination.cos();
        let sin_i = inclination.sin();
        
        // Velocity in orbital plane
        let velocity = SVector::new(
            T::zero(),
            v_circular * cos_i,
            v_circular * sin_i,
        );
        
        self.bodies.push(BodyState::new(id, position, velocity, mass));
    }
    
    /// Initialize Lagrange point L4/L5 Trojan orbit
    pub fn add_lagrange_trojan(&mut self, id: u64, primary_radius: T, lagrange_point: LagrangePoint, mass: T) {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let two = T::one() + T::one();
        let three = two + T::one();
        let six = three + three;
        
        // L4 leads by 60°, L5 trails by 60°
        let angle = match lagrange_point {
            LagrangePoint::L4 => pi / three,  // 60 degrees
            LagrangePoint::L5 => -pi / three,
        };
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        let position = SVector::new(
            primary_radius * cos_a,
            primary_radius * sin_a,
            T::zero(),
        );
        
        // Same orbital speed as primary, rotated by angle
        let v_orbital = (self.integrator.system.mu / primary_radius).sqrt();
        
        let velocity = SVector::new(
            -v_orbital * sin_a,
            v_orbital * cos_a,
            T::zero(),
        );
        
        self.bodies.push(BodyState::new(id, position, velocity, mass));
    }
    
    /// Compute total system energy (kinetic + potential)
    pub fn compute_total_energy(&self) -> T {
        let mut energy = T::zero();
        let two = T::one() + T::one();
        
        // Kinetic energy: Σ ½mv²
        for body in &self.bodies {
            let v_sq = body.velocity.dot(&body.velocity);
            energy = energy + body.mass * v_sq / two;
        }
        
        // Potential energy: -Σ GMm/r (central body)
        for body in &self.bodies {
            let r_sq = body.position.dot(&body.position);
            if r_sq > T::zero() {
                let r = r_sq.sqrt();
                energy = energy - self.integrator.system.mu * body.mass / r;
            }
        }
        
        // Mutual potential energy: -Σ Gm₁m₂/r₁₂
        for i in 0..self.bodies.len() {
            for j in (i+1)..self.bodies.len() {
                let r_vec = self.bodies[j].position - self.bodies[i].position;
                let r_sq = r_vec.dot(&r_vec);
                if r_sq > T::zero() {
                    let r = r_sq.sqrt();
                    energy = energy - self.bodies[i].mass * self.bodies[j].mass / r;
                }
            }
        }
        
        energy
    }
    
    /// Run simulation for n_steps, tracking energy conservation
    pub fn simulate(&mut self, n_steps: usize, dt: T) -> Result<SimulationStats<T>, NBodyError> {
        let initial_energy = self.compute_total_energy();
        let mut max_energy_error = T::zero();
        
        for _step in 0..n_steps {
            self.integrator.step(&mut self.bodies, dt)?;
            
            // Track energy conservation
            let current_energy = self.compute_total_energy();
            let energy_error = (current_energy - initial_energy).abs();
            let rel_error = if initial_energy.abs() > T::zero() {
                energy_error / initial_energy.abs()
            } else {
                energy_error
            };
            
            if rel_error > max_energy_error {
                max_energy_error = rel_error;
            }
        }
        
        Ok(SimulationStats {
            initial_energy,
            final_energy: self.compute_total_energy(),
            max_relative_energy_error: max_energy_error,
            body_count: self.bodies.len(),
        })
    }
    
    /// Detect close approaches that could cause collisions
    pub fn detect_close_approaches(&self, min_distance: T) -> Vec<(u64, u64, T)> {
        let mut approaches = Vec::new();
        
        for i in 0..self.bodies.len() {
            for j in (i+1)..self.bodies.len() {
                let r_vec = self.bodies[j].position - self.bodies[i].position;
                let distance = r_vec.dot(&r_vec).sqrt();
                
                if distance < min_distance {
                    approaches.push((self.bodies[i].id, self.bodies[j].id, distance));
                }
            }
        }

        
        approaches
    }
}

/// Lagrange points for orbital resonances
#[derive(Debug, Clone, Copy)]
pub enum LagrangePoint {
    L4,  // Leading triangular point
    L5,  // Trailing triangular point
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct SimulationStats<T> {
    pub initial_energy: T,
    pub final_energy: T,
    pub max_relative_energy_error: T,
    pub body_count: usize,
}

/// Orbital resonance calculator
pub struct ResonanceCalculator<T> {
    mu: T,
}

impl<T: Float + Copy + Zero> ResonanceCalculator<T> {
    pub fn new(mu: T) -> Self {
        Self { mu }
    }
    
    /// Compute orbital period from semi-major axis: T = 2π√(a³/μ)
    pub fn orbital_period(&self, semi_major_axis: T) -> T {
        let two = T::one() + T::one();
        let three = two + T::one();
        let six = three + three;
        
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let pi_two = two * pi;
        
        let a_cubed = semi_major_axis * semi_major_axis * semi_major_axis;
        (pi_two * (a_cubed / self.mu).sqrt()).sqrt() * (a_cubed / self.mu).sqrt() / (a_cubed / self.mu).sqrt()
    }
    
    /// Check if two orbits are in p:q resonance
    pub fn check_resonance(&self, a1: T, a2: T, p: u32, q: u32, tolerance: T) -> bool {
        let t1 = self.orbital_period(a1);
        let t2 = self.orbital_period(a2);
        
        if t1 <= T::zero() || t2 <= T::zero() {
            return false;
        }
        
        let ratio = t1 / t2;
        let target_ratio = T::from(p as f64).unwrap() / T::from(q as f64).unwrap();
        
        let diff = (ratio - target_ratio).abs();
        diff < tolerance
    }
    
    /// Find resonant semi-major axis for given primary
    pub fn find_resonant_axis(&self, a_primary: T, p: u32, q: u32) -> T {
        // From Kepler's third law: (a/a')³ = (T/T')² = (p/q)²
        let p_f64 = T::from(p as f64).unwrap();
        let q_f64 = T::from(q as f64).unwrap();
        
        let two = T::one() + T::one();
        let three = two + T::one();
        
        // a_resonant = a_primary * (p/q)^(2/3)
        let ratio = p_f64 / q_f64;
        let exponent = two / three;
        
        a_primary * ratio.powf(exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_leapfrog_energy_conservation() {
        type F = f64;
        let mu = F::from(3.986e14).unwrap();  // Earth's μ
        let system = GravitationalSystem::point_mass(mu);
        let mut sim = DysonSwarmSimulator::new(system);
        
        // Add satellite in LEO
        let radius = F::from(6.778e6).unwrap();  // 400 km altitude
        sim.add_circular_orbit(1, radius, F::from(1000.0).unwrap(), F::from(0.0).unwrap());
        
        // Run for 1000 orbits
        let period = F::from(5550.0).unwrap();  // ~92 minutes
        let dt = period / F::from(100.0).unwrap();
        let n_steps = 100_000;
        
        let stats = sim.simulate(n_steps, dt).unwrap();
        
        // Symplectic integrator should maintain bounded energy error
        assert!(stats.max_relative_energy_error < F::from(0.01).unwrap());
    }
    
    #[test]
    fn test_resonance_detection() {
        type F = f64;
        let mu = F::from(1.327e20).unwrap();  // Sun's μ
        let calc = ResonanceCalculator::new(mu);
        
        // Earth at 1 AU
        let earth_a = F::from(1.496e11).unwrap();
        
        // Check 2:1 resonance (asteroid belt Kirkwood gap)
        let resonant_a = calc.find_resonant_axis(earth_a, 1, 2);
        
        // Should be at ~0.63 AU
        assert!(resonant_a < earth_a);
        assert!(calc.check_resonance(earth_a, resonant_a, 2, 1, F::from(0.01).unwrap()));
    }
}
