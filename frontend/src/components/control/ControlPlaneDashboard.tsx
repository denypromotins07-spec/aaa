'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getNexusRPC, CommandCode, RpcMessageType } from '../../hooks/useNexusRPC';
import { useIsEpistemicLockdown } from '../../hooks/useRealityAnchor';

interface SliderConfig {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  command: CommandCode;
  payloadKey: string;
  unit?: string;
}

export default function ControlPlaneDashboard() {
  const isEpistemicLockdown = useIsEpistemicLockdown();
  const [isConnected, setIsConnected] = useState(false);
  const [pendingSync, setPendingSync] = useState<Set<string>>(new Set());
  const rpcRef = useRef(getNexusRPC());
  
  const sliders: SliderConfig[] = [
    {
      id: 'risk-aversion',
      label: 'Stage 15 Market Making Risk Aversion (γ)',
      value: 0.5,
      min: 0.01,
      max: 5.0,
      step: 0.01,
      command: CommandCode.CMD_SET_RISK_AVERSION,
      payloadKey: 'gamma',
      unit: '',
    },
    {
      id: 'lyapunov-bounds',
      label: 'Stage 19 Safe RL Lyapunov Bounds',
      value: 1.0,
      min: 0.1,
      max: 10.0,
      step: 0.1,
      command: CommandCode.CMD_SET_LYAPUNOV_BOUNDS,
      payloadKey: 'bounds',
      unit: 'σ',
    },
  ];

  const [sliderValues, setSliderValues] = useState<Record<string, number>>({
    'risk-aversion': 0.5,
    'lyapunov-bounds': 1.0,
  });

  const [preTradeLimits, setPreTradeLimits] = useState({
    maxNotional: 1000000,
    maxGross: 5000000,
    maxNet: 2000000,
  });

  useEffect(() => {
    const rpc = rpcRef.current;
    
    rpc.setOnAck((seqId, success) => {
      // Find which slider was acked and clear pending state
      setPendingSync((prev) => {
        const next = new Set(prev);
        // Clear all pending (simplified - in prod would track per-command)
        next.clear();
        return next;
      });
      
      if (!success) {
        console.warn('[ControlPlane] Command NACK received');
      }
    });

    rpc.setOnError((error) => {
      console.error('[ControlPlane] RPC Error:', error);
      setIsConnected(false);
    });

    // Connect to RPC endpoint
    rpc.connect('ws://localhost:8080/ws/rpc')
      .then(() => setIsConnected(true))
      .catch((err) => {
        console.error('[ControlPlane] Connection failed:', err);
        setIsConnected(false);
      });

    return () => {
      rpc.disconnect();
    };
  }, []);

  const handleSliderChange = useCallback(async (slider: SliderConfig, newValue: number) => {
    // CRITICAL: Block all RPC commands during epistemic lockdown
    if (isEpistemicLockdown) {
      console.warn('[ControlPlane] Command blocked - Epistemic Humility Mode active');
      return;
    }

    setSliderValues((prev) => ({ ...prev, [slider.id]: newValue }));
    setPendingSync((prev) => new Set(prev).add(slider.id));

    try {
      const rpc = rpcRef.current;
      const payload = { [slider.payloadKey]: newValue };
      
      await rpc.sendCommand(slider.command, payload);
      // Pending state cleared in onAck callback
    } catch (error) {
      console.error(`[ControlPlane] Failed to send ${slider.label}:`, error);
      setPendingSync((prev) => {
        const next = new Set(prev);
        next.delete(slider.id);
        return next;
      });
    }
  }, [isEpistemicLockdown]);

  const handlePreTradeLimitsUpdate = useCallback(async () => {
    // CRITICAL: Block all RPC commands during epistemic lockdown
    if (isEpistemicLockdown) {
      console.warn('[ControlPlane] Pre-trade limits update blocked - Epistemic Humility Mode active');
      return;
    }

    setPendingSync((prev) => new Set(prev).add('pre-trade'));
    
    try {
      await rpcRef.current.setPreTradeLimits(preTradeLimits);
    } catch (error) {
      console.error('[ControlPlane] Failed to update pre-trade limits:', error);
      setPendingSync((prev) => {
        const next = new Set(prev);
        next.delete('pre-trade');
        return next;
      });
    }
  }, [preTradeLimits, isEpistemicLockdown]);

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-mono text-cyan-400 tracking-widest uppercase drop-shadow-[0_0_10px_rgba(0,255,255,0.8)]">
          Strategy Control Plane
        </h2>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono ${
          isConnected 
            ? 'bg-green-900/30 text-green-400 border border-green-500/50' 
            : 'bg-red-900/30 text-red-400 border border-red-500/50'
        }`}>
          <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
          {isConnected ? 'RPC CONNECTED' : 'DISCONNECTED'}
        </div>
      </div>

      {/* Sliders Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {sliders.map((slider) => {
          const isPending = pendingSync.has(slider.id);
          
          return (
            <div 
              key={slider.id}
              className={`relative p-4 rounded-lg border transition-all duration-300 ${
                isPending 
                  ? 'border-yellow-500/50 bg-yellow-900/10' 
                  : 'border-cyan-500/30 bg-gray-900/40'
              } backdrop-blur-sm`}
            >
              {isPending && (
                <div className="absolute top-2 right-2 text-yellow-400 text-xs font-mono animate-pulse">
                  PENDING SYNC...
                </div>
              )}
              
              <label className="block text-sm font-mono text-gray-300 mb-3">
                {slider.label}
              </label>
              
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={sliderValues[slider.id]}
                  onChange={(e) => handleSliderChange(slider, parseFloat(e.target.value))}
                  className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
                <span className="w-20 text-right font-mono text-cyan-400">
                  {sliderValues[slider.id].toFixed(2)}{slider.unit}
                </span>
              </div>
              
              <div className="flex justify-between text-xs font-mono text-gray-500 mt-1">
                <span>{slider.min}{slider.unit}</span>
                <span>{slider.max}{slider.unit}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pre-Trade Risk Limits */}
      <div className="p-4 rounded-lg border border-cyan-500/30 bg-gray-900/40 backdrop-blur-sm">
        <h3 className="text-lg font-mono text-cyan-400 mb-4">Stage 5 Pre-Trade Risk Limits</h3>
        
        <div className="grid grid-cols-3 gap-4 mb-4">
          {[
            { key: 'maxNotional', label: 'Max Notional', suffix: '$' },
            { key: 'maxGross', label: 'Max Gross', suffix: '$' },
            { key: 'maxNet', label: 'Max Net', suffix: '$' },
          ].map((field) => (
            <div key={field.key}>
              <label className="block text-xs font-mono text-gray-400 mb-1">{field.label}</label>
              <input
                type="number"
                value={preTradeLimits[field.key as keyof typeof preTradeLimits]}
                onChange={(e) => setPreTradeLimits((prev) => ({
                  ...prev,
                  [field.key]: parseInt(e.target.value, 10) || 0,
                }))}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-cyan-400 font-mono text-sm focus:border-cyan-400 focus:outline-none"
              />
            </div>
          ))}
        </div>
        
        <button
          onClick={handlePreTradeLimitsUpdate}
          disabled={pendingSync.has('pre-trade')}
          className={`px-4 py-2 font-mono text-sm rounded transition-all ${
            pendingSync.has('pre-trade')
              ? 'bg-yellow-600/50 text-yellow-200 cursor-not-allowed'
              : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-[0_0_15px_rgba(0,255,255,0.4)]'
          }`}
        >
          {pendingSync.has('pre-trade') ? 'SYNCING...' : 'UPDATE LIMITS'}
        </button>
      </div>
    </div>
  );
}
