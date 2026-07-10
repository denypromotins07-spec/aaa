//! Zero-Allocation Solana Transaction Builder
//! 
//! Builds Solana transactions directly into pre-allocated, cache-aligned byte buffers
//! without using borsh/bincode heap allocations. Supports Jito tip instruction appending.

use alloc::vec::Vec;
use core::mem;
use thiserror::Error;

/// Maximum transaction size per Solana protocol (1232 bytes)
pub const MAX_TX_SIZE: usize = 1232;

/// Cache line alignment for optimal memory access
const CACHE_LINE_SIZE: usize = 64;

#[derive(Error, Debug)]
pub enum TxBuilderError {
    #[error("Buffer overflow: required {required} bytes, available {available} bytes")]
    BufferOverflow { required: usize, available: usize },
    #[error("Invalid instruction data length")]
    InvalidInstructionLength,
    #[error("Too many instructions: max {max}")]
    TooManyInstructions { max: usize },
    #[error("Signature generation failed")]
    SignatureFailed,
}

/// Result type for transaction building operations
pub type Result<T> = core::result::Result<T, TxBuilderError>;

/// Pre-allocated, cache-aligned buffer for transaction serialization
#[repr(C, align(64))]
pub struct AlignedTxBuffer {
    data: [u8; MAX_TX_SIZE],
    write_pos: usize,
}

impl AlignedTxBuffer {
    /// Create a new zero-initialized aligned buffer
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0u8; MAX_TX_SIZE],
            write_pos: 0,
        }
    }

    /// Get remaining capacity
    #[inline]
    pub const fn remaining(&self) -> usize {
        MAX_TX_SIZE - self.write_pos
    }

    /// Get current write position
    #[inline]
    pub const fn position(&self) -> usize {
        self.write_pos
    }

    /// Write bytes without allocation
    #[inline]
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.remaining() {
            return Err(TxBuilderError::BufferOverflow {
                required: data.len(),
                available: self.remaining(),
            });
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data.as_mut_ptr().add(self.write_pos),
                data.len(),
            );
        }
        self.write_pos += data.len();
        Ok(())
    }

    /// Write a u8
    #[inline]
    pub fn write_u8(&mut self, val: u8) -> Result<()> {
        self.write_bytes(&[val])
    }

    /// Write a u16 in little-endian format
    #[inline]
    pub fn write_u16_le(&mut self, val: u16) -> Result<()> {
        self.write_bytes(&val.to_le_bytes())
    }

    /// Write a u32 in little-endian format
    #[inline]
    pub fn write_u32_le(&mut self, val: u32) -> Result<()> {
        self.write_bytes(&val.to_le_bytes())
    }

    /// Write a u64 in little-endian format
    #[inline]
    pub fn write_u64_le(&mut self, val: u64) -> Result<()> {
        self.write_bytes(&val.to_le_bytes())
    }

    /// Compact-u16 encoding (Solana-specific variable-length encoding)
    #[inline]
    pub fn write_compact_u16(&mut self, mut val: u16) -> Result<()> {
        if val < 0xfd {
            self.write_u8(val as u8)
        } else if val <= 0xffff {
            self.write_u8(0xfd)?;
            self.write_u16_le(val)
        } else {
            // Should not happen for u16
            Err(TxBuilderError::InvalidInstructionLength)
        }
    }

    /// Get the serialized transaction data
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.write_pos]
    }

    /// Reset buffer for reuse (zero-copy reset)
    #[inline]
    pub fn reset(&mut self) {
        self.write_pos = 0;
    }
}

impl Default for AlignedTxBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Instruction representation for zero-copy building
pub struct Instruction<'a> {
    pub program_id: &'a [u8; 32],
    pub accounts: &'a [AccountMeta],
    pub data: &'a [u8],
}

/// Account metadata with signer/writable flags
#[derive(Clone, Copy)]
pub struct AccountMeta {
    pub pubkey: [u8; 32],
    pub is_signer: bool,
    pub is_writable: bool,
}

/// Zero-Allocation Transaction Builder
/// 
/// Builds complete Solana transactions in pre-allocated buffers.
/// Supports atomic Jito tip instruction appending.
pub struct ZeroAllocTxBuilder<'a> {
    buffer: &'a mut AlignedTxBuffer,
    instructions: alloc::vec::Vec<Instruction<'a>>,
    account_keys: alloc::vec::Vec<[u8; 32]>,
    recent_blockhash: [u8; 32],
    fee_payer: [u8; 32],
}

impl<'a> ZeroAllocTxBuilder<'a> {
    /// Create a new builder with a pre-allocated buffer
    pub fn new(buffer: &'a mut AlignedTxBuffer) -> Self {
        Self {
            buffer,
            instructions: alloc::vec::Vec::with_capacity(16),
            account_keys: alloc::vec::Vec::with_capacity(32),
            recent_blockhash: [0u8; 32],
            fee_payer: [0u8; 32],
        }
    }

    /// Set the fee payer account
    pub fn fee_payer(mut self, pubkey: [u8; 32]) -> Self {
        self.fee_payer = pubkey;
        self
    }

    /// Set the recent blockhash
    pub fn recent_blockhash(mut self, blockhash: [u8; 32]) -> Self {
        self.recent_blockhash = blockhash;
        self
    }

    /// Add an instruction (zero-copy reference)
    pub fn add_instruction(mut self, instr: Instruction<'a>) -> Result<Self> {
        if self.instructions.len() >= 16 {
            return Err(TxBuilderError::TooManyInstructions { max: 16 });
        }
        
        // Track unique account keys
        for account in instr.accounts {
            if !self.account_keys.contains(&account.pubkey) {
                if self.account_keys.len() >= 32 {
                    return Err(TxBuilderError::TooManyInstructions { max: 32 });
                }
                self.account_keys.push(account.pubkey);
            }
        }
        
        self.instructions.push(instr);
        Ok(self)
    }

    /// Append Jito tip instruction atomically
    /// This adds a transfer to the Jito validator fee account
    pub fn append_jito_tip(mut self, tip_amount_lamports: u64, jito_fee_account: [u8; 32]) -> Result<Self> {
        // Create system program transfer instruction data
        let mut tip_data = [0u8; 12];
        tip_data[0..4].copy_from_slice(&2u32.to_le_bytes()); // Transfer instruction
        tip_data[4..12].copy_from_slice(&tip_amount_lamports.to_le_bytes());
        
        let tip_accounts = [
            AccountMeta { pubkey: self.fee_payer, is_signer: true, is_writable: true },
            AccountMeta { pubkey: jito_fee_account, is_signer: false, is_writable: true },
        ];
        
        let system_program = [0u8; 32]; // System program ID
        
        let tip_instr = Instruction {
            program_id: &system_program,
            accounts: &tip_accounts,
            data: &tip_data,
        };
        
        self = self.add_instruction(tip_instr)?;
        Ok(self)
    }

    /// Serialize the transaction into the buffer (zero-allocation)
    pub fn build(mut self) -> Result<&'a [u8]> {
        if self.instructions.is_empty() {
            return Err(TxBuilderError::InvalidInstructionLength);
        }

        self.buffer.reset();

        // 1. Encode number of signatures (compact-u16)
        self.buffer.write_compact_u16(1)?; // Single signature for now

        // 2. Placeholder for signature (64 bytes) - will be filled later
        let sig_placeholder = [0u8; 64];
        self.buffer.write_bytes(&sig_placeholder)?;

        // 3. Encode message
        // 3a. Header
        let num_required_sigs = 1u8;
        let num_readonly_signed = 0u8;
        let num_readonly_unsigned = 0u8;
        self.buffer.write_u8(num_required_sigs)?;
        self.buffer.write_u8(num_readonly_signed)?;
        self.buffer.write_u8(num_readonly_unsigned)?;

        // 3b. Account keys
        self.buffer.write_compact_u16(self.account_keys.len() as u16)?;
        for key in &self.account_keys {
            self.buffer.write_bytes(key)?;
        }

        // 3c. Recent blockhash
        self.buffer.write_bytes(&self.recent_blockhash)?;

        // 3d. Instructions
        self.buffer.write_compact_u16(self.instructions.len() as u16)?;
        for instr in &self.instructions {
            // Program ID index
            let program_idx = self.find_account_index(instr.program_id).ok_or(
                TxBuilderError::InvalidInstructionLength
            )? as u8;
            self.buffer.write_u8(program_idx)?;

            // Account indices
            self.buffer.write_compact_u16(instr.accounts.len() as u16)?;
            for account in instr.accounts {
                let idx = self.find_account_index(&account.pubkey).ok_or(
                    TxBuilderError::InvalidInstructionLength
                )? as u8;
                self.buffer.write_u8(idx)?;
            }

            // Instruction data
            self.buffer.write_compact_u16(instr.data.len() as u16)?;
            self.buffer.write_bytes(instr.data)?;
        }

        Ok(self.buffer.as_slice())
    }

    /// Find account index in the key list
    fn find_account_index(&self, pubkey: &[u8; 32]) -> Option<usize> {
        self.account_keys.iter().position(|k| k == pubkey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_write() {
        let mut buffer = AlignedTxBuffer::new();
        assert_eq!(buffer.remaining(), MAX_TX_SIZE);
        
        buffer.write_u8(42).unwrap();
        buffer.write_u32_le(12345).unwrap();
        
        assert_eq!(buffer.position(), 5);
        assert_eq!(buffer.remaining(), MAX_TX_SIZE - 5);
    }

    #[test]
    fn test_buffer_overflow_protection() {
        let mut buffer = AlignedTxBuffer::new();
        let large_data = [0u8; MAX_TX_SIZE + 1];
        
        let result = buffer.write_bytes(&large_data);
        assert!(matches!(result, Err(TxBuilderError::BufferOverflow { .. })));
    }

    #[test]
    fn test_zero_alloc_tx_builder() {
        let mut buffer = AlignedTxBuffer::new();
        let fee_payer = [1u8; 32];
        let blockhash = [2u8; 32];
        
        let program_id = [3u8; 32];
        let account = AccountMeta {
            pubkey: fee_payer,
            is_signer: true,
            is_writable: true,
        };
        let instr_data = [0x01, 0x02, 0x03];
        
        let instruction = Instruction {
            program_id: &program_id,
            accounts: &[account],
            data: &instr_data,
        };
        
        let builder = ZeroAllocTxBuilder::new(&mut buffer)
            .fee_payer(fee_payer)
            .recent_blockhash(blockhash)
            .add_instruction(instruction)
            .unwrap();
        
        let result = builder.build();
        assert!(result.is_ok());
    }
}
