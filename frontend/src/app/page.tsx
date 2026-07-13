'use client';

import DashboardShell from '@/components/layout/DashboardShell';
import { WebGLHeatmap } from '@/components/orderbook/WebGLHeatmap';
import { MicroPriceTape } from '@/components/orderbook/MicroPriceTape';
import { useNexusSocket } from '@/hooks/useNexusSocket';
import { useConnectionStatus, useHealth } from '@/store/nexusStore';
import { useEffect } from 'react';

export default function Home() {
  const { registerCanvasCallback, sendControlMessage } = useNexusSocket();
  const isConnected = useConnectionStatus();
  const health = useHealth();

  // Dispatch heatmap data updates via custom event
  useEffect(() => {
    const dispatchHeatmapData = (data: unknown) => {
      window.dispatchEvent(new CustomEvent('heatmap-data', { detail: data }));
    };

    registerCanvasCallback(dispatchHeatmapData);
  }, [registerCanvasCallback]);

  return (
    <DashboardShell>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              Orderbook Telemetry
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              Real-time L2/L3 visualization via WebGL
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className={`px-3 py-1.5 rounded-full text-xs font-mono border ${
              isConnected 
                ? 'border-neon-green/30 bg-neon-green/10 text-neon-green' 
                : 'border-neon-red/30 bg-neon-red/10 text-neon-red'
            }`}>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </div>
          </div>
        </div>

        {/* Main Visualization Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Orderbook Heatmap - Takes 2/3 width */}
          <div className="lg:col-span-2 glass-panel rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse" />
                L2 Orderbook Heatmap
              </h3>
              <span className="text-xs text-gray-500 font-mono">BTC-USD</span>
            </div>
            
            <div className="aspect-video w-full">
              <WebGLHeatmap 
                className="rounded-lg overflow-hidden"
                width={800}
                height={450}
              />
            </div>
            
            {/* Legend */}
            <div className="mt-4 flex items-center gap-4 text-xs text-gray-500">
              <span>Volume Intensity:</span>
              <div className="flex items-center gap-1">
                <div className="w-4 h-2 rounded bg-[rgb(0,25,77)]" />
                <div className="w-4 h-2 rounded bg-[rgb(0,245,255)]" />
                <div className="w-4 h-2 rounded bg-[rgb(255,0,255)]" />
                <div className="w-4 h-2 rounded bg-[rgb(255,255,0)]" />
              </div>
              <span className="font-mono">LOW → HIGH</span>
            </div>
          </div>

          {/* Micro Price Tape - Takes 1/3 width */}
          <div className="glass-panel rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-neon-magenta animate-pulse" />
                Trade Flow
              </h3>
              <span className="text-xs text-gray-500 font-mono">
                {health.latencyUs < 100 ? '🟢' : '🟡'} {health.latencyUs}μs
              </span>
            </div>
            
            <div className="w-full" style={{ height: '200px' }}>
              <MicroPriceTape 
                className="rounded-lg overflow-hidden"
                width={400}
                height={200}
              />
            </div>

            {/* Stats */}
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Best Bid</span>
                <span className="font-mono text-neon-green">--</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Best Ask</span>
                <span className="font-mono text-neon-red">--</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Spread</span>
                <span className="font-mono text-gray-300">--</span>
              </div>
            </div>
          </div>
        </div>

        {/* System Metrics Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard 
            label="Latency" 
            value={`${health.latencyUs}μs`}
            status={health.latencyUs < 100 ? 'good' : 'warning'}
          />
          <MetricCard 
            label="PnL" 
            value={health.pnlUsd >= 0 ? `+$${health.pnlUsd.toFixed(2)}` : `-$${Math.abs(health.pnlUsd).toFixed(2)}`}
            status={health.pnlUsd >= 0 ? 'good' : 'bad'}
          />
          <MetricCard 
            label="CPU Usage" 
            value={`${health.cpuUsage.toFixed(1)}%`}
            status={health.cpuUsage < 70 ? 'good' : 'warning'}
          />
          <MetricCard 
            label="Memory" 
            value={`${(health.memoryMb / 1024).toFixed(1)} GB`}
            status="neutral"
          />
        </div>

        {/* Control Panel Placeholder */}
        <div className="glass-panel rounded-xl p-6 border border-white/5">
          <h3 className="text-sm font-medium text-gray-300 mb-4">
            Bot Controls
          </h3>
          <div className="flex gap-4">
            <button 
              onClick={() => sendControlMessage('StartBot')}
              className="px-6 py-2 bg-neon-green/20 hover:bg-neon-green/30 text-neon-green rounded-lg text-sm font-medium transition-colors border border-neon-green/30"
            >
              ▶ START BOT
            </button>
            <button 
              onClick={() => sendControlMessage('StopBot')}
              className="px-6 py-2 bg-neon-red/20 hover:bg-neon-red/30 text-neon-red rounded-lg text-sm font-medium transition-colors border border-neon-red/30"
            >
              ⏹ STOP BOT
            </button>
            <button 
              onClick={() => sendControlMessage('PauseBot')}
              className="px-6 py-2 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 rounded-lg text-sm font-medium transition-colors border border-yellow-500/30"
            >
              ⏸ PAUSE
            </button>
          </div>
        </div>
      </div>
    </DashboardShell>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  status: 'good' | 'warning' | 'bad' | 'neutral';
}

function MetricCard({ label, value, status }: MetricCardProps) {
  const statusColors = {
    good: 'text-neon-green',
    warning: 'text-yellow-400',
    bad: 'text-neon-red',
    neutral: 'text-gray-300',
  };

  return (
    <div className="glass-panel rounded-lg p-4 border border-white/5">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className={`text-lg font-mono ${statusColors[status]}`}>
        {value}
      </div>
    </div>
  );
}
