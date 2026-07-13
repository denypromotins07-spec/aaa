'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface DashboardShellProps {
  children: React.ReactNode;
}

const navItems = [
  { name: 'Telemetry', href: '/', icon: '◈' },
  { name: 'Alpha Signals', href: '/alpha', icon: 'α' },
  { name: 'Risk Matrix', href: '/risk', icon: 'Δ' },
  { name: 'Hardware', href: '/hardware', icon: '⬡' },
  { name: 'Omega Core', href: '/omega', icon: 'Ω' },
];

export default function DashboardShell({ children }: DashboardShellProps) {
  const pathname = usePathname();
  const [currentTime, setCurrentTime] = useState<Date>(new Date());

  // Update clock every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="flex h-screen bg-obsidian overflow-hidden">
      {/* Sidebar Navigation */}
      <aside className="w-64 glass-panel border-r border-glass-border flex flex-col">
        {/* Logo / Brand */}
        <div className="p-6 border-b border-glass-border">
          <h1 className="text-xl font-mono font-bold tracking-wider">
            <span className="text-neon-cyan">NEXUS</span>
            <span className="text-neon-magenta">-OMEGA</span>
          </h1>
          <p className="text-xs text-gray-500 font-mono mt-1">
            GOD-MODE COMMAND CENTER
          </p>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`
                  flex items-center gap-3 px-4 py-3 rounded-lg font-mono text-sm
                  transition-all duration-200
                  ${isActive 
                    ? 'bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30' 
                    : 'text-gray-400 hover:text-gray-100 hover:bg-obsidian-light/50'
                  }
                `}
              >
                <span className="text-lg">{item.icon}</span>
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Bottom Info */}
        <div className="p-4 border-t border-glass-border">
          <div className="glass-panel rounded p-3">
            <div className="text-xs text-gray-500 font-mono">SESSION TIME</div>
            <div className="text-sm font-mono text-neon-cyan mt-1">
              {currentTime.toISOString().split('T')[1].split('.')[0]} UTC
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar - Global System Health */}
        <header className="h-16 glass-panel border-b border-glass-border flex items-center justify-between px-6">
          <div className="flex items-center gap-6">
            <h2 className="font-mono text-sm text-gray-400">
              STAGE <span className="text-neon-cyan">1/5</span>
            </h2>
            
            {/* Quick Stats */}
            <div className="flex items-center gap-4 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="text-gray-500">LAT:</span>
                <span className="text-neon-green">-- μs</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-500">PnL:</span>
                <span className="text-neon-green">--</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-500">SWARM:</span>
                <span className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-neon-green animate-pulse" />
                  <span className="text-neon-green">ONLINE</span>
                </span>
              </div>
            </div>
          </div>

          {/* Right side of top bar */}
          <div className="flex items-center gap-4">
            <div className="font-mono text-xs text-gray-500">
              BACKEND: <span className="text-yellow-500">STANDBY</span>
            </div>
            <button className="px-3 py-1.5 text-xs font-mono rounded border border-neon-cyan/30 text-neon-cyan hover:bg-neon-cyan/10 transition-colors">
              SYSTEM CONFIG
            </button>
          </div>
        </header>

        {/* Page Content */}
        {children}
      </div>
    </div>
  );
}
