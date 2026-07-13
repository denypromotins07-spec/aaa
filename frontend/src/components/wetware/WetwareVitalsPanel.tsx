'use client';

import React, { useRef, useEffect } from 'react';

interface WetwareVitalsPanelProps {
  phLevel: number;
  perfusionRate: number;
  temperature: number;
  oxygenLevel: number;
  glucoseLevel: number;
}

export function WetwareVitalsPanel({
  phLevel,
  perfusionRate,
  temperature,
  oxygenLevel,
  glucoseLevel,
}: WetwareVitalsPanelProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-4 bg-[#0a0a0c]/80 backdrop-blur-md rounded-lg border border-cyan-900/30">
      <VitalCard
        label="pH Level"
        value={phLevel.toFixed(2)}
        unit=""
        normalRange={[7.35, 7.45]}
        currentValue={phLevel}
        color="#00ffff"
      />
      <VitalCard
        label="Perfusion Rate"
        value={perfusionRate.toFixed(1)}
        unit="mL/min"
        normalRange={[0.5, 2.0]}
        currentValue={perfusionRate}
        color="#ff00ff"
      />
      <VitalCard
        label="Temperature"
        value={temperature.toFixed(2)}
        unit="°C"
        normalRange={[36.5, 37.5]}
        currentValue={temperature}
        color="#00ff88"
      />
      <VitalCard
        label="O₂ Level"
        value={oxygenLevel.toFixed(1)}
        unit="mmHg"
        normalRange={[80, 100]}
        currentValue={oxygenLevel}
        color="#ffff00"
      />
      <VitalCard
        label="Glucose"
        value={glucoseLevel.toFixed(1)}
        unit="mg/dL"
        normalRange={[70, 110]}
        currentValue={glucoseLevel}
        color="#ff8800"
      />
      <div className="flex items-center justify-center p-3 bg-cyan-900/20 rounded border border-cyan-800/30">
        <div className="text-center">
          <div className="text-xs text-cyan-600 font-mono uppercase tracking-wider">Status</div>
          <div className="text-lg font-bold text-cyan-400 animate-pulse">OPTIMAL</div>
        </div>
      </div>
    </div>
  );
}

interface VitalCardProps {
  label: string;
  value: string;
  unit: string;
  normalRange: [number, number];
  currentValue: number;
  color: string;
}

function VitalCard({ label, value, unit, normalRange, currentValue, color }: VitalCardProps) {
  const isNormal = currentValue >= normalRange[0] && currentValue <= normalRange[1];
  const isWarning = !isNormal && (
    Math.abs(currentValue - normalRange[0]) < (normalRange[1] - normalRange[0]) * 0.2 ||
    Math.abs(currentValue - normalRange[1]) < (normalRange[1] - normalRange[0]) * 0.2
  );

  return (
    <div 
      className={`p-3 rounded border transition-all duration-300 ${
        isNormal 
          ? 'bg-[#1a1a2e]/50 border-cyan-900/30' 
          : isWarning 
            ? 'bg-amber-900/20 border-amber-700/50 animate-pulse' 
            : 'bg-red-900/20 border-red-700/50 animate-pulse'
      }`}
      style={{ borderColor: isNormal ? undefined : undefined }}
    >
      <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-1">{label}</div>
      <div 
        className="text-xl font-bold font-mono"
        style={{ color: isNormal ? color : isWarning ? '#fbbf24' : '#ef4444' }}
      >
        {value}<span className="text-sm text-gray-400 ml-1">{unit}</span>
      </div>
      {!isNormal && (
        <div className={`text-xs mt-1 font-mono ${isWarning ? 'text-amber-400' : 'text-red-400'}`}>
          ⚠️ OUT OF RANGE
        </div>
      )}
    </div>
  );
}
