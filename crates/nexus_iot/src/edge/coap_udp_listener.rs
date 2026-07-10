//! CoAP UDP Listener for IoT Telemetry
//! 
//! Zero-allocation CoAP packet parser for UDP-based IoT sensors.
//! Implements RFC 7252 with strict validation.

use std::net::{SocketAddr, UdpSocket};
use core::slice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoapError {
    InvalidVersion,
    InvalidTokenLength,
    BufferTooSmall,
    MalformedOption,
    UnsupportedCode,
}

#[derive(Debug, Clone, Copy)]
pub struct CoapPacket<'a> {
    pub version: u8,
    pub token: &'a [u8],
    pub code: u8,
    pub message_id: u16,
    pub payload: &'a [u8],
}

impl<'a> CoapPacket<'a> {
    /// Parse CoAP packet from UDP buffer without allocation
    pub fn parse(buffer: &'a [u8]) -> Result<Self, CoapError> {
        if buffer.len() < 4 {
            return Err(CoapError::BufferTooSmall);
        }

        let first_byte = buffer[0];
        let version = (first_byte >> 6) & 0x03;
        
        // Validate CoAP version (must be 1)
        if version != 1 {
            return Err(CoapError::InvalidVersion);
        }

        let token_length = (first_byte & 0x0F) as usize;
        
        // Token length validation (0-8 bytes valid)
        if token_length > 8 {
            return Err(CoapError::InvalidTokenLength);
        }

        if buffer.len() < 4 + token_length {
            return Err(CoapError::BufferTooSmall);
        }

        let code = buffer[1];
        let message_id = u16::from_be_bytes([buffer[2], buffer[3]]);

        let mut current_pos = 4 + token_length;
        
        // Skip options until payload marker (0xFF) or end
        let mut option_delta = 0u16;
        let mut option_value = 0u16;

        while current_pos < buffer.len() {
            if buffer[current_pos] == 0xFF {
                // Payload marker
                current_pos += 1;
                break;
            }

            if current_pos >= buffer.len() {
                return Err(CoapError::MalformedOption);
            }

            let option_byte = buffer[current_pos];
            let delta = ((option_byte >> 4) & 0x0F) as usize;
            let length = (option_byte & 0x0F) as usize;

            current_pos += 1;

            // Handle extended delta/length
            let actual_delta = match delta {
                0..=12 => delta as u16,
                13 => {
                    if current_pos >= buffer.len() {
                        return Err(CoapError::MalformedOption);
                    }
                    option_delta = buffer[current_pos] as u16 + 13;
                    current_pos += 1;
                    option_delta
                },
                14 => {
                    if current_pos + 1 >= buffer.len() {
                        return Err(CoapError::MalformedOption);
                    }
                    option_delta = u16::from_be_bytes([buffer[current_pos], buffer[current_pos + 1]]) + 269;
                    current_pos += 2;
                    option_delta
                },
                _ => return Err(CoapError::MalformedOption),
            };

            let actual_length = match length {
                0..=12 => length,
                13 => {
                    if current_pos >= buffer.len() {
                        return Err(CoapError::MalformedOption);
                    }
                    let ext_len = buffer[current_pos] as usize + 13;
                    current_pos += 1;
                    ext_len
                },
                14 => {
                    if current_pos + 1 >= buffer.len() {
                        return Err(CoapError::MalformedOption);
                    }
                    let ext_len = u16::from_be_bytes([buffer[current_pos], buffer[current_pos + 1]]) as usize + 269;
                    current_pos += 2;
                    ext_len
                },
                _ => return Err(CoapError::MalformedOption),
            };

            // Skip option value
            current_pos += actual_length;
            
            if current_pos > buffer.len() {
                return Err(CoapError::MalformedOption);
            }
        }

        let token = unsafe {
            slice::from_raw_parts(buffer.as_ptr().add(4), token_length)
        };

        let payload = if current_pos < buffer.len() {
            unsafe {
                slice::from_raw_parts(buffer.as_ptr().add(current_pos), buffer.len() - current_pos)
            }
        } else {
            &[]
        };

        Ok(CoapPacket {
            version,
            token,
            code,
            message_id,
            payload,
        })
    }
}

/// CoAP UDP Listener with zero-copy reception
pub struct CoapUdpListener {
    socket: UdpSocket,
    buffer: Vec<u8>,
}

impl CoapUdpListener {
    pub fn bind(addr: SocketAddr) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        // Pre-allocate buffer for maximum CoAP packet size (typically 1152 bytes for MTU)
        let buffer = vec![0u8; 1500];
        
        Ok(Self { socket, buffer })
    }

    /// Receive and parse CoAP packet without allocation in hot path
    pub fn receive(&mut self) -> Result<(CoapPacket, SocketAddr), CoapError> {
        let (len, addr) = self.socket.recv_from(&mut self.buffer)
            .map_err(|e| CoapError::BufferTooSmall)?;

        let packet = CoapPacket::parse(&self.buffer[..len])?;
        Ok((packet, addr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_version() {
        let buffer = [0b11000000, 0x01, 0x00, 0x01]; // Version 3 (invalid)
        let result = CoapPacket::parse(&buffer);
        assert_eq!(result, Err(CoapError::InvalidVersion));
    }

    #[test]
    fn test_valid_coap() {
        let mut buffer = vec![0b01000001, 0x01, 0x00, 0x01]; // Version 1, token len 1
        buffer.push(0xAB); // Token
        buffer.extend_from_slice(&[0xFF]); // Payload marker
        buffer.extend_from_slice(b"Hello");

        let packet = CoapPacket::parse(&buffer).unwrap();
        assert_eq!(packet.version, 1);
        assert_eq!(packet.token, &[0xAB]);
        assert_eq!(packet.payload, b"Hello");
    }
}
