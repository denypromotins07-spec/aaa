//! Zero-allocation bump allocator for NEXUS-OMEGA.
//!
//! Provides pre-allocated memory pools to avoid heap allocations in hot paths.

/// Simple bump allocator for fixed-size allocations.
pub struct BumpAllocator {
    buffer: Vec<u8>,
    offset: usize,
}

impl BumpAllocator {
    /// Create new allocator with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Allocate memory without heap allocation (uses pre-allocated buffer)
    pub fn allocate(&mut self, size: usize) -> Option<&mut [u8]> {
        if self.offset + size > self.buffer.len() {
            return None; // Out of memory
        }

        let start = self.offset;
        self.offset += size;
        Some(&mut self.buffer[start..self.offset])
    }

    /// Reset allocator (reuse buffer)
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.offset
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for BumpAllocator {
    fn default() -> Self {
        Self::new(1024 * 1024) // 1MB default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_allocator() {
        let mut alloc = BumpAllocator::new(100);
        
        let buf1 = alloc.allocate(50).unwrap();
        assert_eq!(buf1.len(), 50);
        
        let buf2 = alloc.allocate(30).unwrap();
        assert_eq!(buf2.len(), 30);
        
        // Should have 20 bytes remaining
        assert_eq!(alloc.remaining(), 20);
        
        // This should fail
        assert!(alloc.allocate(30).is_none());
        
        // Reset and try again
        alloc.reset();
        assert_eq!(alloc.remaining(), 100);
        
        let buf3 = alloc.allocate(100).unwrap();
        assert_eq!(buf3.len(), 100);
    }
}
