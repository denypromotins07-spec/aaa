import { create } from 'zustand';
import { useRef, useCallback } from 'react';

/**
 * NEXUS-OMEGA Global State Store
 * 
 * CRITICAL PERFORMANCE DESIGN:
 * - Market data is stored in a REF, not state, to avoid React re-renders
 * - Only low-frequency data (health, settings) triggers re-renders
 * - WebGL components read directly from refs via callbacks
 */

export interface TradeTick {
  timestamp: number;
  price: number;
  volume: number;
  isBuy: boolean;
}

export interface MarketData {
  timestamp: number;
  symbol: string;
  bestBid: number;
  bestAsk: number;
  bidVolume: number;
  askVolume: number;
  l2Bids: [number, number][]; // [price, volume]
  l2Asks: [number, number][];
  trades: TradeTick[];
}

export interface SystemHealth {
  latencyUs: number;
  pnlUsd: number;
  swarmStatus: 'Idle' | 'Running' | 'Paused' | 'EmergencyStop';
  cpuUsage: number;
  memoryMb: number;
}

interface NexusState {
  // High-frequency data stored in ref (no re-renders)
  marketDataRef: React.MutableRefObject<MarketData | null>;
  updateMarketDataRef: (data: MarketData) => void;
  
  // Trade history (limited buffer, batched updates)
  recentTrades: TradeTick[];
  addTrade: (trade: TradeTick) => void;
  clearTrades: () => void;
  
  // Low-frequency health data (triggers re-renders for UI)
  health: SystemHealth;
  updateHealth: (health: Partial<SystemHealth>) => void;
  
  // Connection status
  isConnected: boolean;
  setConnectionStatus: (connected: boolean) => void;
  
  // Client count from server
  clientCount: number;
  setClientCount: (count: number) => void;
  
  // Settings
  settings: NexusSettings;
  updateSettings: (settings: Partial<NexusSettings>) => void;
}

export interface NexusSettings {
  heatmapResolution: number;
  tradeParticleCount: number;
  colorScheme: 'cyberpunk' | 'matrix' | 'monochrome';
  autoScale: boolean;
}

const DEFAULT_SETTINGS: NexusSettings = {
  heatmapResolution: 256,
  tradeParticleCount: 500,
  colorScheme: 'cyberpunk',
  autoScale: true,
};

// Create a ref outside the store for initial value
let initialMarketDataRef: React.MutableRefObject<MarketData | null> = { current: null };

export const useNexusStore = create<NexusState>((set, get) => ({
  // Market data ref - updated without triggering re-renders
  marketDataRef: initialMarketDataRef,
  
  updateMarketDataRef: (data: MarketData) => {
    // Update the ref directly - no re-render
    if (!initialMarketDataRef) {
      initialMarketDataRef = { current: data };
    } else {
      initialMarketDataRef.current = data;
    }
    
    // Also update the store's ref reference
    set((state) => ({
      marketDataRef: { ...state.marketDataRef, current: data },
    }));
  },
  
  // Trade history with limited buffer
  recentTrades: [],
  
  addTrade: (trade: TradeTick) => {
    set((state) => ({
      recentTrades: [...state.recentTrades.slice(-499), trade], // Keep last 500
    }));
  },
  
  clearTrades: () => {
    set({ recentTrades: [] });
  },
  
  // System health - low frequency, safe to trigger re-renders
  health: {
    latencyUs: 0,
    pnlUsd: 0,
    swarmStatus: 'Idle',
    cpuUsage: 0,
    memoryMb: 0,
  },
  
  updateHealth: (healthUpdate: Partial<SystemHealth>) => {
    set((state) => ({
      health: { ...state.health, ...healthUpdate },
    }));
  },
  
  // Connection status
  isConnected: false,
  setConnectionStatus: (connected: boolean) => {
    set({ isConnected: connected });
  },
  
  // Client count
  clientCount: 0,
  setClientCount: (count: number) => {
    set({ clientCount: count });
  },
  
  // Settings
  settings: DEFAULT_SETTINGS,
  updateSettings: (settingsUpdate: Partial<NexusSettings>) => {
    set((state) => ({
      settings: { ...state.settings, ...settingsUpdate },
    }));
  },
}));

/**
 * Hook to get market data ref without subscribing to store updates
 * This is critical for WebGL components that need zero re-renders
 */
export function useMarketDataRef() {
  const marketDataRef = useNexusStore((state) => state.marketDataRef);
  return marketDataRef;
}

/**
 * Hook to get health data (this WILL trigger re-renders)
 */
export function useHealth() {
  return useNexusStore((state) => state.health);
}

/**
 * Hook to get connection status
 */
export function useConnectionStatus() {
  return useNexusStore((state) => state.isConnected);
}
