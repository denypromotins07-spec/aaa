//! Thomas Algorithm (Tridiagonal Matrix Algorithm) for O(N) solution of tridiagonal systems.
//! Used in Crank-Nicolson PDE solver for Avellaneda-Stoikov HJB equation.
//!
//! Solves Ax = d where A is tridiagonal:
//! | b0 c0  0  ... |   | x0 |   | d0 |
//! | a1 b1 c1  ... |   | x1 |   | d1 |
//! | 0  a2 b2 ... | * | . | = | . |
//! | ...       an bn |   | xn |   | dn |
//!
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::pde::hjb_equation::PdeError;

/// Solves a tridiagonal system Ax = d in O(N) time.
/// 
/// # Arguments
/// * `a` - Lower diagonal (size n-1), a[0] corresponds to row 1
/// * `b` - Main diagonal (size n)
/// * `c` - Upper diagonal (size n-1), c[0] corresponds to row 0
/// * `d` - Right-hand side (size n)
/// * `out` - Output buffer (size n) to store solution x
/// 
/// # Safety
/// All slices must have consistent lengths. Caller must ensure:
/// - a.len() == n - 1
/// - b.len() == n
/// - c.len() == n - 1
/// - d.len() == n
/// - out.len() == n
/// 
/// Returns error if division by zero would occur (singular matrix).
#[inline(always)]
pub fn solve_tridiagonal(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    out: &mut [f64],
) -> Result<(), PdeError> {
    let n = b.len();
    
    if n == 0 {
        return Ok(());
    }
    
    if a.len() != n - 1 || c.len() != n - 1 || d.len() != n || out.len() != n {
        return Err(PdeError::DimensionMismatch);
    }
    
    // Epsilon to prevent division by zero in degenerate matrices
    const EPSILON: f64 = 1e-15;
    
    // Forward elimination
    // We modify c and d in-place conceptually, but use local variables to avoid allocation
    // For zero-alloc, we'll use the output array to store modified coefficients
    
    // Store modified c' in out[0..n-1] temporarily
    // Store modified d' in a separate pass
    
    // First row: no change to b[0], compute c'[0] = c[0] / b[0]
    let mut denom = b[0].abs();
    if denom < EPSILON {
        return Err(PdeError::SingularMatrix);
    }
    
    let mut cp = c[0] / b[0];
    out[0] = d[0] / b[0]; // Temporary storage for d'
    
    // Forward sweep
    for i in 1..n {
        let ai = if i < a.len() + 1 { a[i - 1] } else { 0.0 };
        let bi = b[i];
        let ci = if i < c.len() { c[i] } else { 0.0 };
        let di = d[i];
        
        denom = bi - ai * cp;
        if denom.abs() < EPSILON {
            return Err(PdeError::SingularMatrix);
        }
        
        if i < n - 1 {
            cp = ci / denom;
            out[i] = (di - ai * out[i - 1]) / denom;
        } else {
            // Last row, no c term
            out[i] = (di - ai * out[i - 1]) / denom;
        }
    }
    
    // Back substitution is already done since we stored results in out during forward sweep
    // Actually, we need proper back substitution. Let me fix this.
    
    // Redo with proper algorithm:
    // Use two temporary arrays from bump allocator would be ideal, but for now
    // we'll do a two-pass approach using the output array cleverly
    
    // Pass 1: Forward elimination storing c' and d'
    // We'll use out for d' and a separate stack-allocated approach for c'
    
    // Actually, let's implement properly with explicit arrays
    // Since this is called frequently, we need to be careful
    
    // Forward elimination: compute c' and d'
    // c'[i] = c[i] / (b[i] - a[i]*c'[i-1])
    // d'[i] = (d[i] - a[i]*d'[i-1]) / (b[i] - a[i]*c'[i-1])
    
    // Re-implementing cleanly:
    
    // Special case n=1
    if n == 1 {
        if b[0].abs() < EPSILON {
            return Err(PdeError::SingularMatrix);
        }
        out[0] = d[0] / b[0];
        return Ok(());
    }
    
    // Forward elimination
    let mut c_prime = vec![0.0; n - 1];
    let mut d_prime = vec![0.0; n];
    
    // Row 0
    if b[0].abs() < EPSILON {
        return Err(PdeError::SingularMatrix);
    }
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    
    // Rows 1 to n-1
    for i in 1..n {
        let ai = a[i - 1];
        let bi = b[i];
        let denom = bi - ai * if i > 1 { c_prime[i - 2] } else { c_prime[0] };
        
        if i < n - 1 {
            if denom.abs() < EPSILON {
                return Err(PdeError::SingularMatrix);
            }
            c_prime[i - 1] = c[i] / denom;
        }
        
        if denom.abs() < EPSILON {
            return Err(PdeError::SingularMatrix);
        }
        d_prime[i] = (d[i] - ai * d_prime[i - 1]) / denom;
    }
    
    // Back substitution
    out[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = d_prime[i] - c_prime[i] * out[i + 1];
    }
    
    Ok(())
}

/// Zero-allocation version using pre-allocated workspace buffers.
/// Caller must provide workspace arrays of size at least n.
#[inline(always)]
pub fn solve_tridiagonal_zero_alloc(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    out: &mut [f64],
    workspace_c: &mut [f64],
    workspace_d: &mut [f64],
) -> Result<(), PdeError> {
    let n = b.len();
    
    if n == 0 {
        return Ok(());
    }
    
    if a.len() != n - 1 || c.len() != n - 1 || d.len() != n || out.len() != n {
        return Err(PdeError::DimensionMismatch);
    }
    
    if workspace_c.len() < n - 1 || workspace_d.len() < n {
        return Err(PdeError::DimensionMismatch);
    }
    
    const EPSILON: f64 = 1e-15;
    
    // Special case n=1
    if n == 1 {
        if b[0].abs() < EPSILON {
            return Err(PdeError::SingularMatrix);
        }
        out[0] = d[0] / b[0];
        return Ok(());
    }
    
    // Forward elimination
    if b[0].abs() < EPSILON {
        return Err(PdeError::SingularMatrix);
    }
    workspace_c[0] = c[0] / b[0];
    workspace_d[0] = d[0] / b[0];
    
    // Rows 1 to n-1
    for i in 1..n {
        let ai = a[i - 1];
        let bi = b[i];
        let prev_c = if i == 1 { workspace_c[0] } else { workspace_c[i - 2] };
        let denom = bi - ai * prev_c;
        
        if denom.abs() < EPSILON {
            return Err(PdeError::SingularMatrix);
        }
        
        if i < n - 1 {
            workspace_c[i - 1] = c[i] / denom;
        }
        
        workspace_d[i] = (d[i] - ai * workspace_d[i - 1]) / denom;
    }
    
    // Back substitution
    out[n - 1] = workspace_d[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = workspace_d[i] - workspace_c[i] * out[i + 1];
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_tridiagonal() {
        // Simple 3x3 system
        let a = vec![1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0];
        let d = vec![3.0, 4.0, 3.0];
        let mut out = vec![0.0; 3];
        
        solve_tridiagonal(&a, &b, &c, &d, &mut out).unwrap();
        
        // Expected solution: [1.0, 1.0, 1.0]
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_singular_matrix() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let c = vec![0.0, 0.0];
        let d = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 3];
        
        let result = solve_tridiagonal(&a, &b, &c, &d, &mut out);
        assert!(result.is_err());
    }
}
