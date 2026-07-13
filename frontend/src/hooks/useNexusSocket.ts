/**
 * Nexus WebSocket Client Hook
 * 
 * Handles binary MessagePack deserialization and feeds data into Zustand store.
 * CRITICAL: Does NOT trigger React re-renders on every message - only updates
 * refs that WebGL components can read directly.
 */

'use client';

import { useEffect, useRef, useCallback } from 'react';
import { decode, encode } from '@msgpack/msgpack';
import { useNexusStore } from '@/store/nexusStore';

export interface TelemetryFrame {
  timestamp_ns: number;
  symbol: number[]; // [u8; 8]
  bids: [number, number][]; // [(price, volume), ...]
  asks: [number, number][];
  trades: [number, number, number][]; // [(price, volume, side), ...]
  health: {
    latency_us: number;
    ops: number;
    pnl_cents: number;
    active_strategies: number;
    memory_mb: number;
  };
}

export interface NexusSocketOptions {
  url?: string;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

const DEFAULT_OPTIONS: Required<NexusSocketOptions> = {
  url: 'ws://localhost:8080/ws/telemetry',
  autoReconnect: true,
  reconnectInterval: 1000,
  maxReconnectAttempts: 10,
};

export function useNexusSocket(options: NexusSocketOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isConnectingRef = useRef(false);
  const frameCountRef = useRef(0);
  
  // Store setters - used sparingly to avoid re-renders
  const updateTelemetry = useNexusStore((state) => state.updateTelemetry);
  const updateConnectionStatus = useNexusStore((state) => state.updateConnectionStatus);
  const updateSystemHealth = useNexusStore((state) => state.updateSystemHealth);
  
  // Direct ref for latest frame - WebGL reads from this without React state
  const latestFrameRef = useRef<TelemetryFrame | null>(null);
  
  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    isConnectingRef.current = false;
  }, []);
  
  const connect = useCallback(() => {
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    isConnectingRef.current = true;
    
    try {
      const ws = new WebSocket(opts.url);
      ws.binaryType = 'arraybuffer'; // Critical for MessagePack
      
      ws.onopen = () => {
        console.log('[NEXUS] Connected to telemetry server');
        isConnectingRef.current = false;
        reconnectAttemptsRef.current = 0;
        frameCountRef.current = 0;
        updateConnectionStatus('connected');
      };
      
      ws.onmessage = (event) => {
        try {
          // Handle binary MessagePack data
          if (event.data instanceof ArrayBuffer) {
            const decoded = decode(new Uint8Array(event.data)) as unknown;
            
            // Check if it's a telemetry frame envelope
            if (decoded && typeof decoded === 'object' && 'Telemetry' in (decoded as Record<string, unknown>)) {
              const telemetryData = (decoded as Record<string, unknown>).Telemetry as TelemetryFrame;
              
              if (telemetryData) {
                // Update the ref immediately - WebGL reads from this
                latestFrameRef.current = telemetryData;
                frameCountRef.current++;
                
                // Throttled store updates (every 10th frame ~6fps for UI)
                if (frameCountRef.current % 10 === 0) {
                  updateTelemetry(telemetryData);
                  updateSystemHealth(telemetryData.health);
                }
              }
            }
          } else if (typeof event.data === 'string') {
            // JSON control messages
            const data = JSON.parse(event.data);
            console.log('[NEXUS] Control message:', data);
          }
        } catch (error) {
          console.error('[NEXUS] Failed to decode message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('[NEXUS] WebSocket error:', error);
        updateConnectionStatus('error');
      };
      
      ws.onclose = (event) => {
        console.log(`[NEXUS] Disconnected: code=${event.code}, reason=${event.reason}`);
        updateConnectionStatus('disconnected');
        wsRef.current = null;
        isConnectingRef.current = false;
        
        // Auto-reconnect logic
        if (opts.autoReconnect && reconnectAttemptsRef.current < opts.maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          console.log(`[NEXUS] Reconnecting... attempt ${reconnectAttemptsRef.current}/${opts.maxReconnectAttempts}`);
          
          reconnectTimerRef.current = setTimeout(() => {
            connect();
          }, opts.reconnectInterval * reconnectAttemptsRef.current); // Exponential backoff
        }
      };
      
      wsRef.current = ws;
    } catch (error) {
      console.error('[NEXUS] Failed to create WebSocket:', error);
      isConnectingRef.current = false;
      updateConnectionStatus('error');
    }
  }, [opts.url, opts.autoReconnect, opts.reconnectInterval, opts.maxReconnectAttempts, updateConnectionStatus, updateTelemetry, updateSystemHealth]);
  
  // Send control command (JSON)
  const sendCommand = useCallback((command: string, payload?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const msg = { Control: { command, payload } };
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);
  
  // Initial connection
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);
  
  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
    latestFrameRef,
    frameCount: frameCountRef.current,
    sendCommand,
    reconnect: connect,
    disconnect,
  };
}
