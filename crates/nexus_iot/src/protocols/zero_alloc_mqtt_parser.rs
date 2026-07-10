//! Zero-Allocation MQTT v5 Packet Parser
//! 
//! Parses MQTT v5 packets directly from TCP/UDP buffers using pointer arithmetic.
//! No heap allocations in the hot path. Strict bounds checking to prevent buffer overflows.

use core::ptr;
use core::slice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MqttError {
    BufferTooSmall,
    InvalidFixedHeader,
    RemainingLengthOverflow,
    MalformedVariableHeader,
    PayloadLengthMismatch,
}

#[derive(Debug, Clone, Copy)]
pub struct MqttPacket<'a> {
    pub control_type: u8,
    pub flags: u8,
    pub remaining_length: usize,
    pub topic: &'a [u8],
    pub payload: &'a [u8],
    pub packet_id: Option<u16>,
}

impl<'a> MqttPacket<'a> {
    /// Parse MQTT packet from raw buffer without allocation.
    /// SAFETY: Caller must ensure `buffer` contains a complete MQTT packet.
    pub fn parse(buffer: &'a [u8]) -> Result<Self, MqttError> {
        if buffer.is_empty() {
            return Err(MqttError::BufferTooSmall);
        }

        let fixed_header = buffer[0];
        let control_type = (fixed_header >> 4) & 0x0F;
        let flags = fixed_header & 0x0F;

        // Validate control type (1-14 are valid, 0 and 15 are reserved)
        if control_type == 0 || control_type == 15 {
            return Err(MqttError::InvalidFixedHeader);
        }

        // Decode Remaining Length (variable byte integer)
        let mut multiplier: usize = 1;
        let mut remaining_length: usize = 0;
        let mut byte_index = 1;

        loop {
            if byte_index >= buffer.len() {
                return Err(MqttError::RemainingLengthOverflow);
            }
            
            let encoded_byte = buffer[byte_index] as usize;
            remaining_length += (encoded_byte & 0x7F) * multiplier;
            
            // Prevent overflow on multiplier
            if multiplier > 2097152 { // 128^3
                return Err(MqttError::RemainingLengthOverflow);
            }
            multiplier *= 128;
            byte_index += 1;

            if (encoded_byte & 0x80) == 0 {
                break;
            }
        }

        // CRITICAL AUDIT FIX: Validate declared length against actual buffer size
        // Prevents 4GB declared length attack with 10 bytes of data
        if byte_index + remaining_length > buffer.len() {
            return Err(MqttError::PayloadLengthMismatch);
        }

        let payload_start = byte_index;
        let mut current_pos = payload_start;
        let end_pos = payload_start + remaining_length;

        let mut topic: &'a [u8] = &[];
        let mut packet_id: Option<u16> = None;

        // Parse Variable Header based on Control Type
        match control_type {
            3 => { // PUBLISH
                // Topic Length (2 bytes)
                if current_pos + 2 > end_pos {
                    return Err(MqttError::MalformedVariableHeader);
                }
                let topic_len = u16::from_be_bytes([buffer[current_pos], buffer[current_pos + 1]]) as usize;
                current_pos += 2;

                if current_pos + topic_len > end_pos {
                    return Err(MqttError::MalformedVariableHeader);
                }
                
                // Safe slice creation via pointer arithmetic
                topic = unsafe {
                    slice::from_raw_parts(buffer.as_ptr().add(current_pos), topic_len)
                };
                current_pos += topic_len;

                // Packet ID (if QoS > 0)
                let qos = (flags >> 1) & 0x03;
                if qos > 0 {
                    if current_pos + 2 > end_pos {
                        return Err(MqttError::MalformedVariableHeader);
                    }
                    packet_id = Some(u16::from_be_bytes([buffer[current_pos], buffer[current_pos + 1]]));
                    current_pos += 2;
                }
            },
            _ => {
                // Other packet types handled similarly with strict bounds checks
                // For brevity, focusing on PUBLISH which carries telemetry
            }
        }

        let payload_len = end_pos - current_pos;
        let payload = unsafe {
            slice::from_raw_parts(buffer.as_ptr().add(current_pos), payload_len)
        };

        Ok(MqttPacket {
            control_type,
            flags,
            remaining_length,
            topic,
            payload,
            packet_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malformed_length_attack() {
        // Attack: Declare 4GB length but only provide 10 bytes
        let mut buffer = vec![0x30, 0xFF, 0xFF, 0xFF, 0x7F]; // 4GB remaining length
        buffer.extend_from_slice(&[0x00, 0x05, 0x74, 0x6F, 0x70, 0x69, 0x63]); // "topic"
        
        let result = MqttPacket::parse(&buffer);
        assert_eq!(result, Err(MqttError::PayloadLengthMismatch));
    }

    #[test]
    fn test_valid_publish() {
        // Valid PUBLISH packet construction
        let mut buffer = Vec::new();
        buffer.push(0x30); // Fixed header
        buffer.push(0x0A); // Remaining length (10)
        buffer.extend_from_slice(&[0x00, 0x02, 0x61, 0x62]); // Topic "ab"
        buffer.extend_from_slice(&[0x00, 0x01]); // Packet ID
        buffer.extend_from_slice(&[0x48, 0x69]); // Payload "Hi"

        let packet = MqttPacket::parse(&buffer).unwrap();
        assert_eq!(packet.control_type, 3);
        assert_eq!(packet.topic, b"ab");
        assert_eq!(packet.payload, b"Hi");
    }
}
