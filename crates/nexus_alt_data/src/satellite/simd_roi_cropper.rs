//! SIMD-Accelerated Region of Interest Cropper
//! 
//! Uses AVX2/AVX-512 instructions for zero-allocation bounding box extraction
//! from satellite imagery.

use std::arch::x86_64::*;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use crate::satellite::zero_copy_geotiff_stream::{RoiBounds, GeotiffError};

/// Target feature detection for SIMD capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCapability {
    Scalar,
    Sse2,
    Avx2,
    Avx512,
}

impl SimdCapability {
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            SimdCapability::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdCapability::Avx2
        } else if is_x86_feature_detected!("sse2") {
            SimdCapability::Sse2
        } else {
            SimdCapability::Scalar
        }
    }
}

/// Aligned buffer for SIMD operations
/// CRITICAL: Ensures 64-byte alignment for AVX-512 compatibility
pub struct AlignedBuffer {
    ptr: *mut f64,
    layout: Layout,
    len: usize,
    capacity: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Create a new aligned buffer with specified capacity
    /// CRITICAL: Uses custom allocator to guarantee 64-byte alignment
    pub fn with_capacity(capacity: usize) -> Self {
        // Ensure 64-byte alignment for AVX-512
        let align = 64;
        let size = capacity * std::mem::size_of::<f64>();
        
        // Create layout with proper alignment
        let layout = Layout::from_size_align(size, align)
            .expect("Invalid layout parameters");
        
        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                panic!("Failed to allocate aligned memory");
            }
            
            AlignedBuffer {
                ptr: ptr as *mut f64,
                layout,
                len: 0,
                capacity,
            }
        }
    }

    /// Get pointer to aligned data
    #[inline]
    pub fn as_ptr(&self) -> *const f64 {
        self.ptr
    }

    /// Get mutable pointer to aligned data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.ptr
    }

    /// Push value without allocation (assumes capacity available)
    #[inline]
    pub fn push(&mut self, value: f64) {
        if self.len < self.capacity {
            unsafe {
                *self.ptr.add(self.len) = value;
            }
            self.len += 1;
        }
    }

    /// Set value at index without bounds checking
    #[inline]
    pub unsafe fn set_unchecked(&mut self, idx: usize, value: f64) {
        *self.ptr.add(idx) = value;
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset length to zero (reuse buffer)
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Fill with zeros
    pub fn zero_fill(&mut self) {
        unsafe {
            ptr::write_bytes(self.ptr as *mut u8, 0, self.layout.size());
        }
        self.len = 0;
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

/// SIMD-accelerated ROI cropper
pub struct SimdRoiCropper {
    capability: SimdCapability,
}

impl SimdRoiCropper {
    pub fn new() -> Self {
        SimdRoiCropper {
            capability: SimdCapability::detect(),
        }
    }

    /// Extract ROI from source image using SIMD acceleration
    /// 
    /// # Arguments
    /// * `source` - Source image data (must be aligned)
    /// * `source_width` - Width of source image
    /// * `bounds` - ROI bounds to extract
    /// * `output` - Pre-allocated output buffer
    /// 
    /// # Safety
    /// Caller must ensure source and output buffers are properly sized
    pub unsafe fn extract_roi(
        &self,
        source: *const f64,
        source_width: u32,
        bounds: &RoiBounds,
        output: &mut AlignedBuffer,
    ) -> Result<(), GeotiffError> {
        match self.capability {
            SimdCapability::Avx512 => self.extract_roi_avx512(source, source_width, bounds, output),
            SimdCapability::Avx2 => self.extract_roi_avx2(source, source_width, bounds, output),
            SimdCapability::Sse2 => self.extract_roi_sse2(source, source_width, bounds, output),
            SimdCapability::Scalar => self.extract_roi_scalar(source, source_width, bounds, output),
        }
    }

    /// AVX-512 accelerated ROI extraction (512-bit vectors, 8 doubles per vector)
    #[target_feature(enable = "avx512f")]
    unsafe fn extract_roi_avx512(
        &self,
        source: *const f64,
        source_width: u32,
        bounds: &RoiBounds,
        output: &mut AlignedBuffer,
    ) -> Result<(), GeotiffError> {
        let roi_width = bounds.width() as usize;
        let roi_height = bounds.height() as usize;
        
        for row in 0..roi_height {
            let src_row = source.add(
                ((bounds.min_y as usize + row) * source_width as usize + bounds.min_x as usize)
            );
            
            let mut col = 0;
            while col + 7 < roi_width {
                // Load 8 doubles using AVX-512
                let vec = _mm512_loadu_pd(src_row.add(col));
                
                // Store to output
                let out_ptr = output.as_mut_ptr().add(row * roi_width + col);
                _mm512_storeu_pd(out_ptr, vec);
                
                col += 8;
            }
            
            // Handle remaining elements
            for c in col..roi_width {
                output.push(*src_row.add(c));
            }
        }
        
        Ok(())
    }

    /// AVX2 accelerated ROI extraction (256-bit vectors, 4 doubles per vector)
    #[target_feature(enable = "avx2")]
    unsafe fn extract_roi_avx2(
        &self,
        source: *const f64,
        source_width: u32,
        bounds: &RoiBounds,
        output: &mut AlignedBuffer,
    ) -> Result<(), GeotiffError> {
        let roi_width = bounds.width() as usize;
        let roi_height = bounds.height() as usize;
        
        for row in 0..roi_height {
            let src_row = source.add(
                ((bounds.min_y as usize + row) * source_width as usize + bounds.min_x as usize)
            );
            
            let mut col = 0;
            while col + 3 < roi_width {
                // Load 4 doubles using AVX2
                let vec = _mm256_loadu_pd(src_row.add(col));
                
                // Store to output
                let out_ptr = output.as_mut_ptr().add(row * roi_width + col);
                _mm256_storeu_pd(out_ptr, vec);
                
                col += 4;
            }
            
            // Handle remaining elements
            for c in col..roi_width {
                output.push(*src_row.add(c));
            }
        }
        
        Ok(())
    }

    /// SSE2 accelerated ROI extraction (128-bit vectors, 2 doubles per vector)
    #[target_feature(enable = "sse2")]
    unsafe fn extract_roi_sse2(
        &self,
        source: *const f64,
        source_width: u32,
        bounds: &RoiBounds,
        output: &mut AlignedBuffer,
    ) -> Result<(), GeotiffError> {
        let roi_width = bounds.width() as usize;
        let roi_height = bounds.height() as usize;
        
        for row in 0..roi_height {
            let src_row = source.add(
                ((bounds.min_y as usize + row) * source_width as usize + bounds.min_x as usize)
            );
            
            let mut col = 0;
            while col + 1 < roi_width {
                // Load 2 doubles using SSE2
                let vec = _mm_loadu_pd(src_row.add(col));
                
                // Store to output
                let out_ptr = output.as_mut_ptr().add(row * roi_width + col);
                _mm_storeu_pd(out_ptr, vec);
                
                col += 2;
            }
            
            // Handle remaining elements
            for c in col..roi_width {
                output.push(*src_row.add(c));
            }
        }
        
        Ok(())
    }

    /// Scalar fallback ROI extraction
    unsafe fn extract_roi_scalar(
        &self,
        source: *const f64,
        source_width: u32,
        bounds: &RoiBounds,
        output: &mut AlignedBuffer,
    ) -> Result<(), GeotiffError> {
        let roi_width = bounds.width() as usize;
        let roi_height = bounds.height() as usize;
        
        for row in 0..roi_height {
            for col in 0..roi_width {
                let src_idx = (bounds.min_y as usize + row) * source_width as usize 
                            + bounds.min_x as usize + col;
                output.push(*source.add(src_idx));
            }
        }
        
        Ok(())
    }

    /// Batch extract multiple ROIs with automatic SIMD dispatch
    pub fn batch_extract(
        &self,
        source: &[f64],
        source_width: u32,
        bounds_list: &[RoiBounds],
    ) -> Result<Vec<AlignedBuffer>, GeotiffError> {
        let mut results = Vec::with_capacity(bounds_list.len());
        
        for bounds in bounds_list {
            bounds.validate(source_width, (source.len() / source_width as usize) as u32)?;
            
            let roi_size = (bounds.width() * bounds.height()) as usize;
            let mut output = AlignedBuffer::with_capacity(roi_size);
            
            unsafe {
                self.extract_roi(
                    source.as_ptr(),
                    source_width,
                    bounds,
                    &mut output,
                )?;
            }
            
            results.push(output);
        }
        
        Ok(results)
    }
}

impl Default for SimdRoiCropper {
    fn default() -> Self {
        Self::new()
    }
}

/// Bounding box calculator for industrial facilities
pub struct FacilityBboxCalculator;

impl FacilityBboxCalculator {
    /// Calculate bounding box for oil storage tanks based on known coordinates
    pub fn calculate_tank_bbox(
        center_lat: f64,
        center_lon: f64,
        tank_radius_meters: f64,
        pixels_per_meter: f64,
    ) -> RoiBounds {
        let radius_pixels = (tank_radius_meters * pixels_per_meter) as u32;
        
        // Convert lat/lon to pixel coordinates (simplified)
        let center_x = (center_lon * 111320.0 * pixels_per_meter) as u32;
        let center_y = (center_lat * 111320.0 * pixels_per_meter) as u32;
        
        RoiBounds::new(
            center_x.saturating_sub(radius_pixels),
            center_y.saturating_sub(radius_pixels),
            center_x + radius_pixels,
            center_y + radius_pixels,
        )
    }

    /// Calculate bounding box for port terminal
    pub fn calculate_port_bbox(
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
        pixels_per_degree: f64,
    ) -> RoiBounds {
        let min_x = (min_lon * pixels_per_degree) as u32;
        let min_y = (min_lat * pixels_per_degree) as u32;
        let max_x = (max_lon * pixels_per_degree) as u32;
        let max_y = (max_lat * pixels_per_degree) as u32;
        
        RoiBounds::new(min_x, min_y, max_x, max_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capability_detection() {
        let cap = SimdCapability::detect();
        assert!(cap >= SimdCapability::Scalar);
    }

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer = AlignedBuffer::with_capacity(1024);
        assert!(buffer.capacity >= 1024);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_facility_bbox_calculation() {
        let bbox = FacilityBboxCalculator::calculate_tank_bbox(
            35.9848, -97.3942, 50.0, 0.5
        );
        assert!(bbox.width() > 0);
        assert!(bbox.height() > 0);
    }
}
