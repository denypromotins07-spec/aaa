/**
 * Nexus Global State Store (Zustand)
 * 
 * CRITICAL: This store is used ONLY for low-frequency UI updates.
 * High-frequency telemetry data is stored in refs and read directly by WebGL.
 * The store updates at most 6fps to avoid React re-render storms.
 */

import { create } from 'zustand';
import { TelemetryFrame } from '@/hooks/useNexusSocket';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface SystemHealth {
  latency_us: number;
  ops: number;
  pnl_cents: number;
  active_strategies: number;
  memory_mb: number;
}

interface NexusState {
  // Connection state
  connectionStatus: ConnectionStatus;
  
  // Throttled telemetry for UI display (not for rendering)
  lastTelemetry: TelemetryFrame | null;
  systemHealth: SystemHealth | null;
  
  // UI state
  selectedSymbol: string;
  timeRange: '1m' | '5m' | '15m' | '1h' | '1d';
  
  // Actions - designed to minimize re-renders
  updateConnectionStatus: (status: ConnectionStatus) => void;
  updateTelemetry: (frame: TelemetryFrame) => void;
  updateSystemHealth: (health: SystemHealth) => void;
  setSelectedSymbol: (symbol: string) => void;
  setTimeRange: (range: '1m' | '5m' | '15m' | '1h' | '1d') => void;
  
  // Selectors for components that need stable references
  getLatestBids: () => [number, number][];
  getLatestAsks: () => [number, number][];
  getLatestTrades: () => [number, number, number][];
}

const defaultHealth: SystemHealth = {
  latency_us: 0,
  ops: 0,
  pnl_cents: 0,
  active_strategies: 0,
  memory_mb: 0,
};

export const useNexusStore = create<NexusState>((set, get) => ({
  // Initial state
  connectionStatus: 'disconnected',
  lastTelemetry: null,
  systemHealth: null,
  selectedSymbol: 'BTCUSD',
  timeRange: '1m',
  
  // Actions
  updateConnectionStatus: (status) => {
    set({ connectionStatus: status });
  },
  
  updateTelemetry: (frame) => {
    // Only update - UI components subscribe selectively
    set({ lastTelemetry: frame });
  },
  
  updateSystemHealth: (health) => {
    set({ systemHealth: health });
  },
  
  setSelectedSymbol: (symbol) => {
    set({ selectedSymbol: symbol });
  },
  
  setTimeRange: (range) => {
    set({ timeRange: range });
  },
  
  // Selectors - return stable arrays even when telemetry is null
  getLatestBids: () => {
    const telemetry = get().lastTelemetry;
    return telemetry?.bids ?? [];
  },
  
  getLatestAsks: () => {
    const telemetry = get().lastTelemetry;
    return telemetry?.asks ?? [];
  },
  
  getLatestTrades: () => {
    const telemetry = get().lastTelemetry;
    return telemetry?.trades ?? [];
  },
}));

// Selector hooks for optimized re-renders
export const useConnectionStatus = () => 
  useNexusStore((state) => state.connectionStatus);

export const useSystemHealth = () => 
  useNexusStore((state) => state.systemHealth);

export const useSelectedSymbol = () => 
  useNexusStore((state) => state.selectedSymbol);
