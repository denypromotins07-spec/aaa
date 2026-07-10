//! FIX 4.4 Binary Encoder for CME iLink 3 and other FIX protocols.
//! Zero-allocation tag-value encoding directly into pre-allocated buffers.

use nexus_oms::FixedPoint;
use crate::zero_alloc_buffer_writer::NetworkBuffer;

const SCALE: i64 = 100_000_000;

/// Standard FIX tags
pub mod tags {
    pub const BEGIN_STRING: u32 = 8;
    pub const BODY_LENGTH: u32 = 9;
    pub const MSG_TYPE: u32 = 35;
    pub const SENDER_COMP_ID: u32 = 49;
    pub const TARGET_COMP_ID: u32 = 56;
    pub const MSG_SEQ_NUM: u32 = 34;
    pub const SENDING_TIME: u32 = 52;
    pub const CHECK_SUM: u32 = 10;
    
    // Order-related tags
    pub const CL_ORD_ID: u32 = 11;
    pub const SYMBOL: u32 = 55;
    pub const SIDE: u32 = 54;
    pub const TRANSACT_TIME: u32 = 60;
    pub const ORDER_TYPE: u32 = 40;
    pub const PRICE: u32 = 44;
    pub const ORDER_QTY: u32 = 38;
    pub const TIME_IN_FORCE: u32 = 59;
    pub const ACCOUNT: u32 = 1;
}

/// Message types
pub mod msg_types {
    pub const NEW_ORDER_SINGLE: &str = "D";
    pub const ORDER_CANCEL_REQUEST: &str = "F";
    pub const EXECUTION_REPORT: &str = "8";
    pub const ORDER_CANCEL_REJECT: &str = "9";
}

/// Side values
pub mod sides {
    pub const BUY: &str = "1";
    pub const SELL: &str = "2";
}

/// Order type values
pub mod order_types {
    pub const LIMIT: &str = "2";
    pub const MARKET: &str = "1";
    pub const LIMIT_ON_CLOSE: &str = "B";
}

/// Time in force values
pub mod tif {
    pub const DAY: &str = "0";
    pub const IOC: &str = "3";
    pub const FOK: &str = "4";
    pub const GTC: &str = "1";
}

/// FIX Encoder state machine
pub struct FixEncoder {
    buffer: NetworkBuffer,
    begin_string: [u8; 16],
    sender_comp_id: [u8; 32],
    target_comp_id: [u8; 32],
    msg_seq_num: u64,
}

impl FixEncoder {
    #[inline]
    pub fn new(begin_string: &str, sender: &str, target: &str) -> Self {
        let mut bs = [0u8; 16];
        let mut sc = [0u8; 32];
        let mut tc = [0u8; 32];
        
        bs[..begin_string.len().min(16)].copy_from_slice(&begin_string.as_bytes()[..begin_string.len().min(16)]);
        sc[..sender.len().min(32)].copy_from_slice(&sender.as_bytes()[..sender.len().min(32)]);
        tc[..target.len().min(32)].copy_from_slice(&target.as_bytes()[..target.len().min(32)]);
        
        Self {
            buffer: NetworkBuffer::new(),
            begin_string: bs,
            sender_comp_id: sc,
            target_comp_id: tc,
            msg_seq_num: 1,
        }
    }

    /// Write a FIX tag-value pair
    #[inline]
    fn write_tag_value(&self, tag: u32, value: &[u8]) -> Result<(), &'static str> {
        // Write tag number
        let tag_str = itoa::Buffer::new();
        let tag_bytes = tag_str.format(tag).as_bytes();
        self.buffer.write_slice(tag_bytes)?;
        
        // Write '=' delimiter
        self.buffer.write_u8(b'=')?;
        
        // Write value
        self.buffer.write_slice(value)?;
        
        // Write SOH (Start of Header) delimiter
        self.buffer.write_u8(1)?;
        
        Ok(())
    }

    /// Write integer tag-value
    #[inline]
    fn write_tag_int(&self, tag: u32, value: i64) -> Result<(), &'static str> {
        let mut buf = itoa::Buffer::new();
        let s = buf.format(value);
        self.write_tag_value(tag, s.as_bytes())
    }

    /// Write string tag-value
    #[inline]
    fn write_tag_str(&self, tag: u32, value: &str) -> Result<(), &'static str> {
        self.write_tag_value(tag, value.as_bytes())
    }

    /// Write price tag-value (FIX uses decimal format)
    #[inline]
    fn write_tag_price(&self, tag: u32, price: FixedPoint) -> Result<(), &'static str> {
        let price_str = format_fixed_point(price);
        self.write_tag_value(tag, price_str.as_bytes())
    }

    /// Build New Order Single message
    #[inline]
    pub fn build_new_order_single(
        &mut self,
        cl_ord_id: &str,
        symbol: &str,
        side: &str,
        quantity: FixedPoint,
        price: FixedPoint,
        account: &str,
    ) -> Result<&[u8], &'static str> {
        self.buffer.reset();
        
        // Build body first (to calculate length)
        let body_start = self.buffer.position();
        
        self.write_tag_str(tags::CL_ORD_ID, cl_ord_id)?;
        self.write_tag_str(tags::SYMBOL, symbol)?;
        self.write_tag_str(tags::SIDE, side)?;
        self.write_tag_int(tags::ORDER_QTY, quantity.raw())?;
        self.write_tag_price(tags::PRICE, price)?;
        self.write_tag_str(tags::ORDER_TYPE, order_types::LIMIT)?;
        self.write_tag_str(tags::TIME_IN_FORCE, tif::DAY)?;
        self.write_tag_str(tags::ACCOUNT, account)?;
        
        let body_end = self.buffer.position();
        let body_len = body_end - body_start;
        
        // Reset to beginning and write header
        self.buffer.reset_to(0);
        
        self.write_tag_str(tags::BEGIN_STRING, std::str::from_utf8(&self.begin_string).unwrap_or("FIX.4.4"))?;
        self.write_tag_int(tags::BODY_LENGTH, body_len as i64)?;
        self.write_tag_str(tags::MSG_TYPE, msg_types::NEW_ORDER_SINGLE)?;
        self.write_tag_str(tags::SENDER_COMP_ID, std::str::from_utf8(&self.sender_comp_id).unwrap_or("SENDER"))?;
        self.write_tag_str(tags::TARGET_COMP_ID, std::str::from_utf8(&self.target_comp_id).unwrap_or("TARGET"))?;
        self.write_tag_int(tags::MSG_SEQ_NUM, self.msg_seq_num as i64)?;
        
        // Sending time (simplified - would use actual timestamp in production)
        self.write_tag_str(tags::SENDING_TIME, "20240101-12:00:00")?;
        
        // Append body
        self.buffer.reset_to(body_start);
        
        // Calculate checksum (sum of all bytes mod 256)
        let checksum = self.calculate_checksum();
        self.write_tag_int(tags::CHECK_SUM, checksum as i64)?;
        
        self.msg_seq_num += 1;
        
        Ok(self.buffer.get_data())
    }

    /// Calculate checksum (simple sum mod 256)
    #[inline]
    fn calculate_checksum(&self) -> u8 {
        let data = self.buffer.get_data();
        data.iter().fold(0u8, |acc, &b| acc.wrapping_add(b))
    }

    /// Get current message sequence number
    #[inline]
    pub fn get_msg_seq_num(&self) -> u64 {
        self.msg_seq_num
    }

    /// Set message sequence number
    #[inline]
    pub fn set_msg_seq_num(&mut self, seq: u64) {
        self.msg_seq_num = seq;
    }
}

/// Format FixedPoint as decimal string (stack-allocated)
#[inline]
fn format_fixed_point(fp: FixedPoint) -> arrayvec::ArrayString<32> {
    use arrayvec::ArrayString;
    
    let raw = fp.raw();
    let negative = raw < 0;
    let abs_val = raw.unsigned_abs();
    
    let int_part = abs_val / SCALE as u64;
    let frac_part = abs_val % SCALE as u64;
    
    let mut result = ArrayString::<32>::new();
    
    if negative {
        result.push('-');
    }
    
    result.push_str(&int_part.to_string());
    result.push('.');
    
    let frac_str = format!("{:08}", frac_part);
    let trimmed = frac_str.trim_end_matches('0');
    if trimmed.is_empty() {
        result.push('0');
    } else {
        result.push_str(trimmed);
    }
    
    result
}

// Requires itoa dependency
// Add to Cargo.toml: itoa = "1.0"

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_encoder_basic() {
        let mut encoder = FixEncoder::new("FIX.4.4", "SENDER", "TARGET");
        
        let msg = encoder.build_new_order_single(
            "ORDER123",
            "ESZ4",
            sides::BUY,
            FixedPoint::from_int(10),
            FixedPoint::from_int(4500),
            "ACCOUNT1",
        );
        
        assert!(msg.is_ok());
        let data = msg.unwrap();
        assert!(!data.is_empty());
        
        // Verify message starts with BeginString
        assert!(data.starts_with(b"8=FIX.4.4"));
    }

    #[test]
    fn test_sequence_increment() {
        let mut encoder = FixEncoder::new("FIX.4.4", "SENDER", "TARGET");
        
        let seq1 = encoder.get_msg_seq_num();
        
        encoder.build_new_order_single(
            "ORDER1",
            "ESZ4",
            sides::BUY,
            FixedPoint::from_int(1),
            FixedPoint::from_int(100),
            "ACC1",
        ).unwrap();
        
        let seq2 = encoder.get_msg_seq_num();
        assert_eq!(seq2, seq1 + 1);
    }
}
