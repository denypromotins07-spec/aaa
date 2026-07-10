//! Zero-Allocation Buffer Writer for network packet serialization.
//! Writes directly into pre-allocated, cache-aligned byte buffers using pointer arithmetic.

use std::sync::atomic::{AtomicU64, Ordering};
use nexus_oms::FixedPoint;

const SCALE: i64 = 100_000_000;

/// Maximum buffer size for a single message
pub const MAX_MESSAGE_SIZE: usize = 4096;

/// Cache-aligned network buffer
#[repr(C, align(64))]
pub struct NetworkBuffer {
    /// Pre-allocated byte array
    data: [u8; MAX_MESSAGE_SIZE],
    /// Current write position
    pos: AtomicU64,
    /// Message start position (for reset)
    start_pos: u64,
}

impl NetworkBuffer {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: [0u8; MAX_MESSAGE_SIZE],
            pos: AtomicU64::new(0),
            start_pos: 0,
        }
    }

    /// Get current write position
    #[inline]
    pub fn position(&self) -> usize {
        self.pos.load(Ordering::Acquire) as usize
    }

    /// Get remaining capacity
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        MAX_MESSAGE_SIZE - self.position()
    }

    /// Check if we can write `n` bytes
    #[inline]
    pub fn can_write(&self, n: usize) -> bool {
        self.remaining_capacity() >= n
    }

    /// Write a single byte
    #[inline]
    pub fn write_u8(&self, val: u8) -> Result<(), &'static str> {
        let pos = self.pos.load(Ordering::Acquire) as usize;
        if pos >= MAX_MESSAGE_SIZE {
            return Err("Buffer overflow");
        }
        
        // SAFETY: We've checked bounds above
        unsafe {
            *self.data.get_unchecked_mut(pos) = val;
        }
        self.pos.store((pos + 1) as u64, Ordering::Release);
        Ok(())
    }

    /// Write u16 in big-endian format
    #[inline]
    pub fn write_u16_be(&self, val: u16) -> Result<(), &'static str> {
        if !self.can_write(2) {
            return Err("Buffer overflow");
        }
        
        let pos = self.pos.load(Ordering::Acquire) as usize;
        let bytes = val.to_be_bytes();
        
        unsafe {
            *self.data.get_unchecked_mut(pos) = bytes[0];
            *self.data.get_unchecked_mut(pos + 1) = bytes[1];
        }
        self.pos.store((pos + 2) as u64, Ordering::Release);
        Ok(())
    }

    /// Write u32 in big-endian format
    #[inline]
    pub fn write_u32_be(&self, val: u32) -> Result<(), &'static str> {
        if !self.can_write(4) {
            return Err("Buffer overflow");
        }
        
        let pos = self.pos.load(Ordering::Acquire) as usize;
        let bytes = val.to_be_bytes();
        
        unsafe {
            *self.data.get_unchecked_mut(pos) = bytes[0];
            *self.data.get_unchecked_mut(pos + 1) = bytes[1];
            *self.data.get_unchecked_mut(pos + 2) = bytes[2];
            *self.data.get_unchecked_mut(pos + 3) = bytes[3];
        }
        self.pos.store((pos + 4) as u64, Ordering::Release);
        Ok(())
    }

    /// Write u64 in big-endian format
    #[inline]
    pub fn write_u64_be(&self, val: u64) -> Result<(), &'static str> {
        if !self.can_write(8) {
            return Err("Buffer overflow");
        }
        
        let pos = self.pos.load(Ordering::Acquire) as usize;
        let bytes = val.to_be_bytes();
        
        unsafe {
            for (i, &b) in bytes.iter().enumerate() {
                *self.data.get_unchecked_mut(pos + i) = b;
            }
        }
        self.pos.store((pos + 8) as u64, Ordering::Release);
        Ok(())
    }

    /// Write FixedPoint price as scaled integer (big-endian)
    #[inline]
    pub fn write_price(&self, price: FixedPoint) -> Result<(), &'static str> {
        // Price is stored as raw i64, convert to u64 for network
        let raw = price.raw();
        let unsigned = if raw < 0 { 0u64 } else { raw as u64 };
        self.write_u64_be(unsigned)
    }

    /// Write FixedPoint quantity as scaled integer (big-endian)
    #[inline]
    pub fn write_quantity(&self, qty: FixedPoint) -> Result<(), &'static str> {
        let raw = qty.raw();
        let unsigned = if raw < 0 { 0u64 } else { raw as u64 };
        self.write_u64_be(unsigned)
    }

    /// Write a byte slice
    #[inline]
    pub fn write_slice(&self, data: &[u8]) -> Result<(), &'static str> {
        if !self.can_write(data.len()) {
            return Err("Buffer overflow");
        }
        
        let pos = self.pos.load(Ordering::Acquire) as usize;
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data.as_mut_ptr().add(pos),
                data.len(),
            );
        }
        self.pos.store((pos + data.len()) as u64, Ordering::Release);
        Ok(())
    }

    /// Write a fixed-size string (padded with spaces or nulls)
    #[inline]
    pub fn write_fixed_str(&self, s: &str, size: usize) -> Result<(), &'static str> {
        if !self.can_write(size) {
            return Err("Buffer overflow");
        }
        
        let pos = self.pos.load(Ordering::Acquire) as usize;
        let bytes = s.as_bytes();
        let copy_len = bytes.len().min(size);
        
        unsafe {
            let dest = self.data.as_mut_ptr().add(pos);
            // Copy string bytes
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, copy_len);
            // Pad with spaces
            for i in copy_len..size {
                *dest.add(i) = b' ';
            }
        }
        self.pos.store((pos + size) as u64, Ordering::Release);
        Ok(())
    }

    /// Reset buffer to start position
    #[inline]
    pub fn reset(&self) {
        self.pos.store(self.start_pos, Ordering::Release);
    }

    /// Reset and set new start position
    #[inline]
    pub fn reset_to(&self, pos: usize) {
        self.start_pos = pos as u64;
        self.pos.store(pos as u64, Ordering::Release);
    }

    /// Get the written data as a slice
    #[inline]
    pub fn get_data(&self) -> &[u8] {
        let len = self.pos.load(Ordering::Acquire) as usize;
        &self.data[..len]
    }

    /// Get the full buffer (for reuse)
    #[inline]
    pub fn get_full_buffer(&self) -> &[u8; MAX_MESSAGE_SIZE] {
        &self.data
    }

    /// Get written length
    #[inline]
    pub fn len(&self) -> usize {
        self.pos.load(Ordering::Acquire) as usize
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for NetworkBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: NetworkBuffer uses atomic operations for thread-safe writes
unsafe impl Send for NetworkBuffer {}
unsafe impl Sync for NetworkBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_u8() {
        let buf = NetworkBuffer::new();
        buf.write_u8(0x42).unwrap();
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.get_data()[0], 0x42);
    }

    #[test]
    fn test_write_u32_be() {
        let buf = NetworkBuffer::new();
        buf.write_u32_be(0x12345678).unwrap();
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.get_data(), &[0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn test_write_price() {
        let buf = NetworkBuffer::new();
        let price = FixedPoint::from_int(100); // 100.00000000
        buf.write_price(price).unwrap();
        assert_eq!(buf.len(), 8);
        
        // Verify the bytes represent 100 * 10^8 in big-endian
        let expected = (100i64 * SCALE) as u64;
        let mut actual = 0u64;
        for (i, &b) in buf.get_data().iter().enumerate() {
            actual |= (b as u64) << (56 - i * 8);
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_buffer_overflow() {
        let buf = NetworkBuffer::new();
        
        // Fill the buffer
        for _ in 0..MAX_MESSAGE_SIZE {
            if buf.write_u8(0).is_err() {
                break;
            }
        }
        
        // Next write should fail
        assert!(buf.write_u8(0).is_err());
    }

    #[test]
    fn test_reset() {
        let buf = NetworkBuffer::new();
        buf.write_u32_be(0x12345678).unwrap();
        assert_eq!(buf.len(), 4);
        
        buf.reset();
        assert_eq!(buf.len(), 0);
        
        // Can write again from start
        buf.write_u8(0xAB).unwrap();
        assert_eq!(buf.get_data()[0], 0xAB);
    }
}
