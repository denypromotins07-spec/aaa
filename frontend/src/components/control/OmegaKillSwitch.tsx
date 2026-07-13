'use client';

import React, { useState, useCallback, useRef } from 'react';
import { getNexusRPC } from '../../hooks/useNexusRPC';

interface OmegaKillSwitchProps {
  onHalt?: () => void;
}

export default function OmegaKillSwitch({ onHalt }: OmegaKillSwitchProps) {
  const [isArmed, setIsArmed] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);
  const rpcRef = useRef(getNexusRPC());

  const startCountdown = useCallback(() => {
    if (!isArmed) {
      setIsArmed(true);
      let count = 3;
      setCountdown(count);
      
      countdownRef.current = setInterval(() => {
        count--;
        if (count <= 0) {
          if (countdownRef.current) {
            clearInterval(countdownRef.current);
            countdownRef.current = null;
          }
          setCountdown(null);
          executeHalt();
        } else {
          setCountdown(count);
        }
      }, 1000);
    }
  }, [isArmed]);

  const cancelCountdown = useCallback(() => {
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
    setIsArmed(false);
    setCountdown(null);
  }, []);

  const executeHalt = useCallback(async () => {
    setIsExecuting(true);
    
    try {
      const rpc = rpcRef.current;
      
      // Send both halt commands in parallel for maximum speed
      await Promise.all([
        rpc.haltExecution(),
        rpc.flattenPortfolio(),
      ]);
      
      console.log('[KillSwitch] HALT commands sent successfully');
      onHalt?.();
    } catch (error) {
      console.error('[KillSwitch] Failed to send HALT commands:', error);
    } finally {
      setIsExecuting(false);
      setIsArmed(false);
    }
  }, [onHalt]);

  React.useEffect(() => {
    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, []);

  const getStatusColor = () => {
    if (isExecuting) return 'from-red-600 to-red-800';
    if (countdown !== null) return 'from-yellow-500 to-orange-600';
    if (isArmed) return 'from-orange-500 to-red-600';
    return 'from-gray-700 to-gray-900';
  };

  const getGlowColor = () => {
    if (isExecuting) return 'rgba(255, 0, 0, 0.8)';
    if (countdown !== null) return 'rgba(255, 200, 0, 0.8)';
    if (isArmed) return 'rgba(255, 100, 0, 0.6)';
    return 'rgba(100, 100, 100, 0.3)';
  };

  return (
    <div className="p-6">
      <h2 className="text-xl font-mono text-red-400 tracking-widest uppercase mb-4 drop-shadow-[0_0_10px_rgba(255,0,0,0.8)]">
        ⚠ OMEGA KILL SWITCH ⚠
      </h2>
      
      <div 
        className={`relative w-full aspect-square max-w-md mx-auto rounded-full bg-gradient-to-br ${getStatusColor()} 
          shadow-[0_0_30px_${getGlowColor()}] border-4 border-red-500/50 
          flex items-center justify-center cursor-pointer transition-all duration-300
          hover:shadow-[0_0_50px_${getGlowColor()}] active:scale-95`}
        onClick={countdown === null && !isExecuting ? startCountdown : undefined}
      >
        {/* Animated rings */}
        {(isArmed || countdown !== null || isExecuting) && (
          <>
            <div className="absolute inset-0 rounded-full border-2 border-red-400 animate-ping opacity-30" />
            <div className="absolute inset-2 rounded-full border border-orange-400 animate-pulse opacity-50" />
          </>
        )}
        
        {/* Center content */}
        <div className="text-center z-10">
          {isExecuting ? (
            <>
              <div className="text-4xl font-black text-white animate-pulse">HALTING</div>
              <div className="text-sm font-mono text-red-200 mt-2">SENDING COMMANDS...</div>
            </>
          ) : countdown !== null ? (
            <>
              <div className="text-6xl font-black text-white animate-pulse">{countdown}</div>
              <div className="text-sm font-mono text-yellow-200 mt-2">CONFIRM TO ABORT</div>
            </>
          ) : isArmed ? (
            <>
              <div className="text-2xl font-black text-white">ARMED</div>
              <div className="text-xs font-mono text-orange-200 mt-2">CLICK TO CONFIRM</div>
            </>
          ) : (
            <>
              <div className="text-2xl font-bold text-gray-300">EMERGENCY</div>
              <div className="text-lg font-black text-red-400 mt-1">STOP ALL</div>
              <div className="text-xs font-mono text-gray-400 mt-2">CLICK TO ARM</div>
            </>
          )}
        </div>
        
        {/* Danger stripes pattern */}
        <div className="absolute inset-0 rounded-full overflow-hidden pointer-events-none opacity-20">
          <div 
            className="w-full h-full"
            style={{
              backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(0,0,0,0.3) 10px, rgba(0,0,0,0.3) 20px)',
            }}
          />
        </div>
      </div>
      
      {/* Warning text */}
      <div className="mt-6 text-center">
        <p className="text-xs font-mono text-red-300/70">
          THIS WILL IMMEDIATELY HALT ALL TRADING ACTIVITY AND FLATTEN ALL POSITIONS
        </p>
        <p className="text-xs font-mono text-gray-500 mt-2">
          CMD_HALT_EXECUTION + CMD_FLATTEN_PORTFOLIO
        </p>
      </div>
      
      {/* Manual cancel button when armed */}
      {isArmed && countdown === null && !isExecuting && (
        <button
          onClick={cancelCountdown}
          className="mt-4 w-full py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 font-mono text-sm rounded border border-gray-600 transition-colors"
        >
          DISARM
        </button>
      )}
    </div>
  );
}
