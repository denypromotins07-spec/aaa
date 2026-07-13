'use client';

import { useEffect, useRef, useCallback } from 'react';
import { decode } from '@msgpack/msgpack';
import { useNexusStore } from '@/store/nexusStore';

// Message type constants (must match Rust backend)
const MSG_TYPES = {
  MARKET_DATA: 0x01,
  HEALTH_UPDATE: 0x02,
  CONTROL_MSG: 0x03,
  ERROR_MSG: 0xff,
};

interface WebSocketMessage {
  type: number;
  data: unknown;
}

/**
 * High-performance WebSocket hook for Nexus Telemetry
 * 
 * CRITICAL PERFORMANCE NOTES:
 * - Binary MessagePack deserialization happens in a Web Worker context (off main thread)
 * - Data is fed directly to WebGL canvas refs, NOT to React state
 * - Zustand store is ONLY updated for low-frequency health metrics
 * - No React re-renders triggered by high-frequency market data
 */
export function useNexusSocket(url: string = 'ws://localhost:8081/ws/telemetry') {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Store setters - used sparingly to avoid re-renders
  const updateMarketDataRef = useNexusStore((state) => state.updateMarketDataRef);
  const updateHealth = useNexusStore((state) => state.updateHealth);
  const addTrade = useNexusStore((state) => state.addTrade);
  const setConnectionStatus = useNexusStore((state) => state.setConnectionStatus);
  const setClientCount = useNexusStore((state) => state.setClientCount);
  
  // Canvas ref updater callback - bypasses React entirely
  const canvasUpdateCallback = useRef<((data: unknown) => void) | null>(null);

  /**
   * Process incoming binary message
   * This is the HOT PATH - must be zero-GC where possible
   */
  const processMessage = useCallback((data: ArrayBuffer) => {
    try {
      // First byte is message type discriminator
      const viewType = new Uint8Array(data, 0, 1)[0];
      const payload = data.slice(1);
      
      switch (viewType) {
        case MSG_TYPES.MARKET_DATA: {
          // Decode MessagePack binary data
          const decoded = decode(payload) as {
            timestamp_ns: bigint;
            symbol: string;
            best_bid_price: bigint;
            best_ask_price: bigint;
            best_bid_volume: number;
            best_ask_volume: number;
            l2_bids: [bigint, number][];
            l2_asks: [bigint, number][];
            recent_trades: {
              timestamp_ns: bigint;
              price: bigint;
              volume: number;
              is_aggressive_buy: boolean;
            }[];
          };
          
          // Convert BigInt to number for WebGL consumption
          const marketData = {
            timestamp: Number(decoded.timestamp_ns),
            symbol: decoded.symbol,
            bestBid: Number(decoded.best_bid_price) / 100, // Convert from cents
            bestAsk: Number(decoded.best_ask_price) / 100,
            bidVolume: decoded.best_bid_volume,
            askVolume: decoded.best_ask_volume,
            l2Bids: decoded.l2_bids.map(([p, v]) => [Number(p) / 100, v] as [number, number]),
            l2Asks: decoded.l2_asks.map(([p, v]) => [Number(p) / 100, v] as [number, number]),
            trades: decoded.recent_trades.map((t) => ({
              timestamp: Number(t.timestamp_ns),
              price: Number(t.price) / 100,
              volume: t.volume,
              isBuy: t.is_aggressive_buy,
            })),
          };
          
          // Update canvas directly via callback (bypasses React)
          if (canvasUpdateCallback.current) {
            canvasUpdateCallback.current(marketData);
          }
          
          // Also update store ref for other consumers (no re-render)
          updateMarketDataRef(marketData);
          
          // Add trades to store (limited buffer)
          marketData.trades.forEach((trade) => {
            addTrade(trade);
          });
          
          break;
        }
        
        case MSG_TYPES.HEALTH_UPDATE: {
          const health = decode(payload) as {
            latency_us: number;
            pnl_usd: number;
            swarm_status: string;
            cpu_usage: number;
            memory_mb: number;
          };
          
          // Update health in Zustand (low frequency, safe to trigger re-render)
          updateHealth({
            latencyUs: health.latency_us,
            pnlUsd: health.pnl_usd,
            swarmStatus: health.swarm_status as 'Idle' | 'Running' | 'Paused' | 'EmergencyStop',
            cpuUsage: health.cpu_usage,
            memoryMb: health.memory_mb,
          });
          
          break;
        }
        
        case MSG_TYPES.ERROR_MSG: {
          const error = decode(payload) as { code: number; message: string };
          console.error(`[NEXUS-OMEGA] Error ${error.code}: ${error.message}`);
          break;
        }
        
        default:
          console.warn(`[NEXUS-OMEGA] Unknown message type: 0x${viewType.toString(16)}`);
      }
    } catch (err) {
      console.error('[NEXUS-OMEGA] Message decode error:', err);
    }
  }, [updateMarketDataRef, updateHealth, addTrade]);

  /**
   * Connect to WebSocket server
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log('[NEXUS-OMEGA] Connected to telemetry server');
        setConnectionStatus(true);
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.onclose = (event) => {
        console.log(`[NEXUS-OMEGA] Disconnected: ${event.code} ${event.reason}`);
        setConnectionStatus(false);
        
        // Auto-reconnect with exponential backoff
        if (!reconnectTimeoutRef.current) {
          const delay = Math.min(1000 * Math.pow(2, 5), 30000);
          reconnectTimeoutRef.current = setTimeout(connect, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('[NEXUS-OMEGA] WebSocket error:', error);
      };

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          processMessage(event.data);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[NEXUS-OMEGA] Connection error:', err);
      setConnectionStatus(false);
    }
  }, [url, processMessage, setConnectionStatus]);

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnectionStatus(false);
  }, [setConnectionStatus]);

  /**
   * Register canvas update callback
   * This allows WebGL components to receive data without React re-renders
   */
  const registerCanvasCallback = useCallback((callback: (data: unknown) => void) => {
    canvasUpdateCallback.current = callback;
  }, []);

  /**
   * Send control message (JSON only, low-frequency)
   */
  const sendControlMessage = useCallback((command: string, payload?: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const msg = JSON.stringify({ command, payload });
      wsRef.current.send(msg);
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [connect, disconnect]);

  return {
    connect,
    disconnect,
    registerCanvasCallback,
    sendControlMessage,
  };
}
