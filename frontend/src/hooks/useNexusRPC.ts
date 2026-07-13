/**
 * NEXUS-OMEGA FRONTEND STAGE 2
 * Module: Nexus RPC Protocol
 * Purpose: Binary MessagePack RPC client with strict schema validation and sequence tracking.
 * Security: Validates packet headers, tracks sequence IDs, prevents command injection.
 */

import { pack, unpack } from '@msgpack/msgpack';

// Binary Protocol Schema
export enum RpcMessageType {
  CMD = 0x01,
  ACK = 0x02,
  ERROR = 0x03,
  HEARTBEAT = 0x04,
}

export enum CommandCode {
  CMD_HALT_EXECUTION = 0x10,
  CMD_FLATTEN_PORTFOLIO = 0x11,
  CMD_SET_RISK_AVERSION = 0x20,
  CMD_SET_LYAPUNOV_BOUNDS = 0x21,
  CMD_SET_PRE_TRADE_LIMITS = 0x22,
  CMD_GET_STATUS = 0x30,
}

interface RpcEnvelope {
  version: number;      // Protocol version (must be 1)
  seqId: number;        // Sequence ID for request/response matching
  msgType: RpcMessageType;
  command?: CommandCode;
  payload?: Record<string, unknown>;
  timestamp: number;    // Unix ms
  checksum: number;     // Simple checksum for integrity
}

const PROTOCOL_VERSION = 1;

function calculateChecksum(envelope: Omit<RpcEnvelope, 'checksum'>): number {
  // Simple XOR-based checksum for demonstration
  // In production, use CRC32 or similar
  const data = JSON.stringify({
    version: envelope.version,
    seqId: envelope.seqId,
    msgType: envelope.msgType,
    command: envelope.command,
    payload: envelope.payload,
    timestamp: envelope.timestamp,
  });
  
  let checksum = 0;
  for (let i = 0; i < data.length; i++) {
    checksum ^= data.charCodeAt(i);
  }
  return checksum & 0xff;
}

function validateEnvelope(envelope: unknown): envelope is RpcEnvelope {
  if (typeof envelope !== 'object' || envelope === null) return false;
  
  const env = envelope as Record<string, unknown>;
  
  // Strict type checking
  if (typeof env.version !== 'number' || env.version !== PROTOCOL_VERSION) {
    console.error('[RPC] Invalid protocol version:', env.version);
    return false;
  }
  
  if (typeof env.seqId !== 'number' || env.seqId < 0) {
    console.error('[RPC] Invalid sequence ID:', env.seqId);
    return false;
  }
  
  if (typeof env.msgType !== 'number' || !(env.msgType in RpcMessageType)) {
    console.error('[RPC] Invalid message type:', env.msgType);
    return false;
  }
  
  if (typeof env.timestamp !== 'number') {
    console.error('[RPC] Missing timestamp');
    return false;
  }
  
  if (typeof env.checksum !== 'number') {
    console.error('[RPC] Missing checksum');
    return false;
  }
  
  // Verify checksum
  const { checksum: receivedChecksum, ...rest } = env;
  const calculatedChecksum = calculateChecksum(rest as Omit<RpcEnvelope, 'checksum'>);
  
  if (receivedChecksum !== calculatedChecksum) {
    console.error('[RPC] Checksum mismatch! Possible tampering detected.');
    return false;
  }
  
  return true;
}

class PendingRequest {
  constructor(
    public seqId: number,
    public resolve: (value: RpcEnvelope) => void,
    public reject: (reason: Error) => void,
    public timeoutId: ReturnType<typeof setTimeout>
  ) {}
}

export class NexusRPCClient {
  private ws: WebSocket | null = null;
  private seqCounter = 0;
  private pendingRequests = new Map<number, PendingRequest>();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  private onAckCallback?: (seqId: number, success: boolean) => void;
  private onErrorCallback?: (error: string) => void;

  connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url, ['binary']);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          console.log('[RPC] Connected to', url);
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onerror = (event) => {
          console.error('[RPC] WebSocket error:', event);
          reject(new Error('WebSocket connection failed'));
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onclose = () => {
          console.log('[RPC] Connection closed');
          this.attemptReconnect(url);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private attemptReconnect(url: string) {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[RPC] Max reconnect attempts reached');
      this.onErrorCallback?.('Connection lost after max retries');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`[RPC] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect(url).catch(console.error);
    }, delay);
  }

  private handleMessage(data: ArrayBuffer) {
    try {
      const decoded = unpack(new Uint8Array(data)) as unknown;
      
      if (!validateEnvelope(decoded)) {
        console.error('[RPC] Invalid envelope received, dropping packet');
        return;
      }

      const envelope = decoded as RpcEnvelope;

      // Handle ACK/ERROR responses
      if (envelope.msgType === RpcMessageType.ACK || envelope.msgType === RpcMessageType.ERROR) {
        const pending = this.pendingRequests.get(envelope.seqId);
        if (pending) {
          clearTimeout(pending.timeoutId);
          this.pendingRequests.delete(envelope.seqId);
          
          if (envelope.msgType === RpcMessageType.ACK) {
            pending.resolve(envelope);
            this.onAckCallback?.(envelope.seqId, true);
          } else {
            const errorMsg = envelope.payload?.error as string || 'Unknown error';
            pending.reject(new Error(errorMsg));
            this.onAckCallback?.(envelope.seqId, false);
          }
        }
      } else if (envelope.msgType === RpcMessageType.HEARTBEAT) {
        // Silently ignore heartbeats
      } else {
        console.warn('[RPC] Unexpected message type:', envelope.msgType);
      }
    } catch (error) {
      console.error('[RPC] Failed to decode message:', error);
    }
  }

  private async sendCommand(command: CommandCode, payload?: Record<string, unknown>): Promise<RpcEnvelope> {
    // Check for epistemic lockdown - block all commands except heartbeat
    if (this.isLockdownActive() && command !== CommandCode.CMD_GET_STATUS) {
      console.warn('[NEXUS RPC] Command blocked due to epistemic humility lockdown');
      throw new Error('EXECUTION_BLOCKED_EPISTEMIC_LOCKDOWN');
    }

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const seqId = ++this.seqCounter;
    const timestamp = Date.now();

    const envelope: Omit<RpcEnvelope, 'checksum'> = {
      version: PROTOCOL_VERSION,
      seqId,
      msgType: RpcMessageType.CMD,
      command,
      payload,
      timestamp,
    };

    const checksum = calculateChecksum(envelope);
    const fullEnvelope: RpcEnvelope = { ...envelope, checksum };

    const binaryData = pack(fullEnvelope);

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(seqId);
        reject(new Error(`Command timeout (seq: ${seqId})`));
      }, 10000); // 10 second timeout

      this.pendingRequests.set(seqId, new PendingRequest(seqId, resolve, reject, timeoutId));

      try {
        this.ws!.send(binaryData);
      } catch (error) {
        clearTimeout(timeoutId);
        this.pendingRequests.delete(seqId);
        reject(error);
      }
    });
  }

  // Public API Methods

  async haltExecution(): Promise<void> {
    await this.sendCommand(CommandCode.CMD_HALT_EXECUTION);
  }

  async flattenPortfolio(): Promise<void> {
    await this.sendCommand(CommandCode.CMD_FLATTEN_PORTFOLIO);
  }

  async setRiskAversion(gamma: number): Promise<void> {
    await this.sendCommand(CommandCode.CMD_SET_RISK_AVERSION, { gamma });
  }

  async setLyapunovBounds(bounds: number): Promise<void> {
    await this.sendCommand(CommandCode.CMD_SET_LYAPUNOV_BOUNDS, { bounds });
  }

  async setPreTradeLimits(limits: { maxNotional: number; maxGross: number; maxNet: number }): Promise<void> {
    await this.sendCommand(CommandCode.CMD_SET_PRE_TRADE_LIMITS, limits);
  }

  async getStatus(): Promise<RpcEnvelope> {
    return this.sendCommand(CommandCode.CMD_GET_STATUS);
  }

  setOnAck(callback: (seqId: number, success: boolean) => void): void {
    this.onAckCallback = callback;
  }

  setOnError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    // Clear all pending requests
    this.pendingRequests.forEach((pending) => {
      clearTimeout(pending.timeoutId);
      pending.reject(new Error('Connection closed'));
    });
    this.pendingRequests.clear();
  }
}

// Singleton instance for app-wide use
let rpcInstance: NexusRPCClient | null = null;

export function getNexusRPC(): NexusRPCClient {
  if (!rpcInstance) {
    rpcInstance = new NexusRPCClient();
  }
  return rpcInstance;
}
