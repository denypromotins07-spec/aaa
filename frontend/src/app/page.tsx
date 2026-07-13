'use client';

import DashboardShell from '@/components/layout/DashboardShell';
import OrderbookHeatmap from '@/components/orderbook/WebGLHeatmap';
import MicroPriceTape from '@/components/orderbook/MicroPriceTape';
import AlphaTopology3D from '@/components/alpha/AlphaTopology3D';
import ControlPlaneDashboard from '@/components/control/ControlPlaneDashboard';
import OmegaKillSwitch from '@/components/control/OmegaKillSwitch';
import PnLWaterfallCanvas from '@/components/risk/PnLWaterfallCanvas';
import CVaRSpeedometer from '@/components/risk/CVaRSpeedometer';
import SiliconThermalHeatmap from '@/components/hardware/SiliconThermalHeatmap';
import PhotonicMeshVisualizer from '@/components/hardware/PhotonicMeshVisualizer';
import SwarmTopologyGraph from '@/components/swarm/SwarmTopologyGraph';
import RaftConsensusTimeline from '@/components/swarm/RaftConsensusTimeline';
import XDPPacketFlow from '@/components/network/XDPPacketFlow';
import DarkPoolSankey from '@/components/network/DarkPoolSankey';
import { useNexusSocket } from '@/hooks/useNexusSocket';
import { useConnectionStatus, useSystemHealth } from '@/store/nexusStore';
import { useState, useEffect } from 'react';
import { AlphaNode } from '@/utils/math/correlationToSpatial';

// Mock alpha data for visualization (in production, this comes from WebSocket)
const MOCK_ALPHAS: AlphaNode[] = [
  { id: 'SMC_01', conviction: 0.85, pnlContribution: 12500, correlationCluster: 1, position: { x: -20, y: 15, z: -10 }, velocity: { x: 0, y: 0, z: 0 } },
  { id: 'SMC_02', conviction: 0.72, pnlContribution: 8900, correlationCluster: 1, position: { x: -15, y: 20, z: -5 }, velocity: { x: 0, y: 0, z: 0 } },
  { id: 'STATARB_01', conviction: 0.45, pnlContribution: -2100, correlationCluster: 2, position: { x: 25, y: -10, z: 15 }, velocity: { x: 0, y: 0, z: 0 } },
  { id: 'STATARB_02', conviction: 0.38, pnlContribution: -1500, correlationCluster: 2, position: { x: 30, y: -5, z: 20 }, velocity: { x: 0, y: 0, z: 0 } },
  { id: 'MACRO_01', conviction: 0.91, pnlContribution: 25000, correlationCluster: 3, position: { x: -10, y: -25, z: 10 }, velocity: { x: 0, y: 0, z: 0 } },
  { id: 'ECONOPHYS_01', conviction: 0.65, pnlContribution: 5600, correlationCluster: 4, position: { x: 15, y: 10, z: 25 }, velocity: { x: 0, y: 0, z: 0 } },
  { id: 'MM_01', conviction: 0.55, pnlContribution: 3200, correlationCluster: 5, position: { x: 5, y: 30, z: -15 }, velocity: { x: 0, y: 0, z: 0 } },
];

export default function Home() {
  const { latestFrameRef, isConnected } = useNexusSocket();
  const connectionStatus = useConnectionStatus();
  const health = useSystemHealth();
  
  // Stage 2 State
  const [isRegimeShift, setIsRegimeShift] = useState(false);
  const [cvarValue, setCvarValue] = useState(0.032);
  const [pnlData, setPnlData] = useState<number[]>([0, 120, 350, 280, 520, 890, 750, 1200, 1450, 1380]);

  // Simulate real-time updates (in production, from WebSocket)
  useEffect(() => {
    const interval = setInterval(() => {
      // Update CVaR with slight random walk
      setCvarValue(prev => Math.max(0.001, Math.min(0.15, prev + (Math.random() - 0.5) * 0.005)));
      
      // Update PnL data
      setPnlData(prev => {
        const lastVal = prev[prev.length - 1] || 0;
        const newVal = lastVal + (Math.random() - 0.45) * 200;
        return [...prev.slice(-49), newVal]; // Keep last 50 points
      });
    }, 500);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <DashboardShell>
      <main className="flex-1 p-6 overflow-auto space-y-6">
        
        {/* SECTION 1: Alpha Topology & Kill Switch (Stage 2) */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[400px]">
          <div className="lg:col-span-3 glass-panel rounded-lg p-4 border border-glass-border relative overflow-hidden">
            <AlphaTopology3D alphas={MOCK_ALPHAS} isRegimeShift={isRegimeShift} />
            
            <button
              onClick={() => setIsRegimeShift(!isRegimeShift)}
              className={`absolute top-4 right-4 px-3 py-1 text-xs font-mono rounded border transition-all ${
                isRegimeShift 
                  ? 'bg-red-900/50 border-red-500 text-red-300' 
                  : 'bg-gray-800/50 border-gray-600 text-gray-400'
              }`}
            >
              {isRegimeShift ? 'REGIME ACTIVE' : 'TOGGLE REGIME'}
            </button>
          </div>
          
          <div className="glass-panel rounded-lg border border-glass-border overflow-hidden">
            <OmegaKillSwitch onHalt={() => console.log('[UI] Halted by user')} />
          </div>
        </div>

        {/* SECTION 2: Orderbook Heatmap & Price Tape (Stage 1) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-cyan mb-4 tracking-wider">
              L2/L3 ORDERBOOK HEATMAP
            </h2>
            <div className="h-[400px] w-full" id="orderbook-canvas-container">
              <OrderbookHeatmap width={800} height={400} latestFrameRef={latestFrameRef} />
            </div>
          </div>
          
          <div className="glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-magenta mb-4 tracking-wider">
              MICRO-PRICE TAPE
            </h2>
            <div className="h-[400px]" id="price-tape-container">
              <MicroPriceTape width={400} height={400} latestFrameRef={latestFrameRef} />
            </div>
          </div>
        </div>

        {/* SECTION 3: Hardware Telemetry (Stage 3) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[350px]">
          <div className="glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-cyan mb-4 tracking-wider">
              SILICON THERMAL MAP :: FPGA
            </h2>
            <SiliconThermalHeatmap width={600} height={300} minTemp={30} maxTemp={95} />
          </div>
          
          <div className="glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-magenta mb-4 tracking-wider">
              PHOTONIC MZI MESH :: STAGE 32
            </h2>
            <PhotonicMeshVisualizer width={600} height={300} />
          </div>
        </div>

        {/* SECTION 4: Swarm Topology (Stage 3) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[400px]">
          <div className="lg:col-span-2 glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-cyan mb-4 tracking-wider">
              SWARM TOPOLOGY :: RAFT CONSENSUS
            </h2>
            <SwarmTopologyGraph width={800} height={350} />
          </div>
          
          <div className="glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-yellow mb-4 tracking-wider">
              RAFT EVENT TIMELINE
            </h2>
            <RaftConsensusTimeline width={350} height={350} />
          </div>
        </div>

        {/* SECTION 5: Network Flow (Stage 3) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[350px]">
          <div className="glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-cyan mb-4 tracking-wider">
              XDP PACKET FLOW :: SMARTNIC
            </h2>
            <XDPPacketFlow width={700} height={300} />
          </div>
          
          <div className="glass-panel rounded-lg p-4 border border-glass-border">
            <h2 className="text-sm font-mono text-neon-green mb-4 tracking-wider">
              DARK POOL ROUTING :: TCA
            </h2>
            <DarkPoolSankey width={700} height={300} />
          </div>
        </div>

        {/* SECTION 6: Control Plane & Risk Surface (Stage 2) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 glass-panel rounded-lg border border-glass-border">
            <ControlPlaneDashboard />
          </div>
          
          <div className="space-y-6">
            <div className="glass-panel rounded-lg p-4 border border-glass-border">
              <h2 className="text-sm font-mono text-neon-red mb-2 tracking-wider text-center">
                STAGE 11 EVT / STAGE 19 CVaR
              </h2>
              <CVaRSpeedometer currentValue={cvarValue} maxValue={0.1} criticalThreshold={0.05} />
            </div>
            
            <div className="glass-panel rounded-lg p-4 border border-glass-border">
              <h2 className="text-sm font-mono text-neon-green mb-2 tracking-wider">
                CUMULATIVE PnL WATERFALL
              </h2>
              <div className="h-[150px]">
                <PnLWaterfallCanvas data={pnlData} width={350} height={150} />
              </div>
            </div>
          </div>
        </div>

        {/* System Health Footer */}
        <div className="glass-panel rounded-lg p-4 border border-glass-border">
          <div className="flex flex-wrap gap-6 justify-between items-center font-mono text-xs">
            <div className="flex items-center gap-4">
              <span className="text-gray-400">LATENCY:</span>
              <span className="text-neon-cyan">{health?.latency_us ?? '--'} μs</span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-gray-400">OPS:</span>
              <span className="text-neon-cyan">{health?.ops ?? '--'}</span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-gray-400">PnL:</span>
              <span className={health && health.pnl_cents >= 0 ? 'text-neon-green' : 'text-neon-red'}>
                {health ? `${(health.pnl_cents / 100).toFixed(2)}` : '$0.00'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-neon-green animate-pulse' :
                connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                connectionStatus === 'error' ? 'bg-neon-red' :
                'bg-gray-500'
              }`} />
              <span className="text-gray-400">NEXUS LINK: {connectionStatus.toUpperCase()}</span>
            </div>
          </div>
        </div>
      </main>
    </DashboardShell>
  );
}
