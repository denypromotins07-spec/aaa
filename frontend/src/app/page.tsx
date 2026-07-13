'use client';

import DashboardShell from '@/components/layout/DashboardShell';
import OrderbookHeatmap from '@/components/orderbook/WebGLHeatmap';
import MicroPriceTape from '@/components/orderbook/MicroPriceTape';
import { useNexusSocket } from '@/hooks/useNexusSocket';
import { useConnectionStatus, useSystemHealth } from '@/store/nexusStore';

export default function Home() {
  const { latestFrameRef, isConnected } = useNexusSocket();
  const connectionStatus = useConnectionStatus();
  const health = useSystemHealth();

  return (
    <DashboardShell>
      <main className="flex-1 p-6 overflow-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          {/* Orderbook Heatmap - Takes 2 columns */}
          <div className="lg:col-span-2 glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-cyan mb-4 tracking-wider">
              L2/L3 ORDERBOOK HEATMAP
            </h2>
            <div className="h-[600px] w-full" id="orderbook-canvas-container">
              <OrderbookHeatmap 
                width={800} 
                height={600} 
                latestFrameRef={latestFrameRef} 
              />
            </div>
          </div>

          {/* Right Panel - System Stats */}
          <div className="space-y-6">
            {/* Micro-Price Tape */}
            <div className="glass-panel rounded-lg p-4 border border-glass-border">
              <h2 className="text-sm font-mono text-neon-magenta mb-4 tracking-wider">
                MICRO-PRICE TAPE
              </h2>
              <div className="h-[200px]" id="price-tape-container">
                <MicroPriceTape
                  width={400}
                  height={200}
                  latestFrameRef={latestFrameRef}
                />
              </div>
            </div>

            {/* System Health */}
            <div className="glass-panel rounded-lg p-4 border border-glass-border">
              <h2 className="text-sm font-mono text-neon-green mb-4 tracking-wider">
                SYSTEM HEALTH
              </h2>
              <div className="space-y-3 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">LATENCY</span>
                  <span className="text-neon-cyan">{health?.latency_us ?? '--'} μs</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">OPS</span>
                  <span className="text-neon-cyan">{health?.ops ?? '--'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">PnL</span>
                  <span className={health && health.pnl_cents >= 0 ? 'text-neon-green' : 'text-neon-red'}>
                    {health ? `${(health.pnl_cents / 100).toFixed(2)}` : '--'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">MEMORY</span>
                  <span className="text-neon-cyan">{health?.memory_mb ?? '--'} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">STRATEGIES</span>
                  <span className="text-neon-cyan">{health?.active_strategies ?? '--'}</span>
                </div>
              </div>
            </div>

            {/* Connection Status */}
            <div className="glass-panel rounded-lg p-4 border border-glass-border">
              <h2 className="text-sm font-mono text-neon-red mb-4 tracking-wider">
                NEXUS LINK
              </h2>
              <div className="flex items-center gap-2">
                <div 
                  className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-neon-green animate-pulse' :
                    connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                    connectionStatus === 'error' ? 'bg-neon-red' :
                    'bg-gray-500'
                  }`} 
                  id="connection-indicator" 
                />
                <span className="font-mono text-xs text-gray-400" id="connection-status">
                  {connectionStatus.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </DashboardShell>
  );
}
