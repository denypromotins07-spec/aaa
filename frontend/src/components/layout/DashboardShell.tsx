'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface SystemHealth {
  latencyUs: number;
  pnlUsd: number;
  swarmStatus: 'Idle' | 'Running' | 'Paused' | 'EmergencyStop';
  cpuUsage: number;
  memoryMb: number;
}

// Message type constants (must match Rust backend)
const MSG_TYPES = {
  MARKET_DATA: 0x01,
  HEALTH_UPDATE: 0x02,
  CONTROL_MSG: 0x03,
  ERROR_MSG: 0xFF,
};

export default function DashboardShell({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [health, setHealth] = useState<SystemHealth>({
    latencyUs: 0,
    pnlUsd: 0,
    swarmStatus: 'Idle',
    cpuUsage: 0,
    memoryMb: 0,
  });
  const [isConnected, setIsConnected] = useState(false);
  const [clientCount, setClientCount] = useState(0);

  // Navigation items
  const navItems = [
    { name: 'Telemetry', href: '/', icon: '📊' },
    { name: 'Alpha Signals', href: '/alpha', icon: '🧬' },
    { name: 'Risk Matrix', href: '/risk', icon: '⚠️' },
    { name: 'Hardware', href: '/hardware', icon: '🔧' },
    { name: 'Omega Core', href: '/omega', icon: '🌀' },
  ];

  // Format numbers with monospace-friendly formatting
  const formatPnL = (value: number) => {
    const formatted = value.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    return value >= 0 ? `+$${formatted}` : `-$${Math.abs(value).toLocaleString()}`;
  };

  const formatLatency = (us: number) => {
    if (us < 1000) return `${us}μs`;
    return `${(us / 1000).toFixed(2)}ms`;
  };

  const getSwarmColor = (status: string) => {
    switch (status) {
      case 'Running': return 'text-neon-green';
      case 'Paused': return 'text-yellow-400';
      case 'EmergencyStop': return 'text-neon-red';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="flex h-screen bg-obsidian overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 glass-panel border-r border-white/5 flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-white/5">
          <h1 className="text-xl font-bold neon-text-cyan tracking-wider">
            NEXUS<span className="text-white">-</span>OMEGA
          </h1>
          <p className="text-xs text-gray-500 mt-1 font-mono">GOD-MODE v1.0</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group ${
                pathname === item.href
                  ? 'bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              <span className="font-medium text-sm">{item.name}</span>
              {pathname === item.href && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-neon-cyan animate-pulse-glow" />
              )}
            </Link>
          ))}
        </nav>

        {/* Connection Status */}
        <div className="p-4 border-t border-white/5">
          <div className="flex items-center gap-2 text-sm">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-neon-green animate-pulse' : 'bg-neon-red'
              }`}
            />
            <span className={isConnected ? 'text-neon-green' : 'text-neon-red'}>
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
          {isConnected && (
            <p className="text-xs text-gray-500 mt-1 font-mono">
              Clients: {clientCount}
            </p>
          )}
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar - System Health */}
        <header className="h-16 glass-panel border-b border-white/5 flex items-center justify-between px-6">
          <div className="flex items-center gap-6">
            {/* Latency */}
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs uppercase tracking-wider">Latency</span>
              <span className={`font-mono text-sm ${health.latencyUs < 100 ? 'text-neon-green' : 'text-yellow-400'}`}>
                {formatLatency(health.latencyUs)}
              </span>
            </div>

            {/* PnL */}
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs uppercase tracking-wider">PnL</span>
              <span className={`font-mono text-sm ${health.pnlUsd >= 0 ? 'text-neon-green' : 'text-neon-red'}`}>
                {formatPnL(health.pnlUsd)}
              </span>
            </div>

            {/* Swarm Status */}
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs uppercase tracking-wider">Swarm</span>
              <span className={`font-mono text-sm ${getSwarmColor(health.swarmStatus)}`}>
                {health.swarmStatus.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Right side - System metrics */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">CPU</span>
              <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-neon-cyan transition-all duration-300"
                  style={{ width: `${Math.min(health.cpuUsage, 100)}%` }}
                />
              </div>
              <span className="font-mono text-xs text-gray-400">{health.cpuUsage.toFixed(1)}%</span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">MEM</span>
              <span className="font-mono text-xs text-gray-400">
                {(health.memoryMb / 1024).toFixed(1)}GB
              </span>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
