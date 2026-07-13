'use client';

import React, { useEffect } from 'react';
import { useRealityAnchor, useIsEpistemicLockdown, RealityAnchorState } from '../../hooks/useRealityAnchor';

interface OntologicalDriftMonitorProps {
  klDivergence: number;
  predictedReality: number[];
  actualReality: number[];
}

export function OntologicalDriftMonitor({ 
  klDivergence,
  predictedReality,
  actualReality 
}: OntologicalDriftMonitorProps) {
  const updateKLDivergence = useRealityAnchor(state => state.updateKLDivergence);
  const realityAnchorState = useRealityAnchor(state => state.realityAnchorState);

  // Update the global store with new KL divergence value
  useEffect(() => {
    updateKLDivergence(klDivergence);
  }, [klDivergence, updateKLDivergence]);

  const getStatusColor = (state: RealityAnchorState) => {
    switch (state) {
      case 'LOCKDOWN': return '#ef4444';
      case 'WARNING': return '#fbbf24';
      case 'IDLE': return '#00ff88';
    }
  };

  const getStatusText = (state: RealityAnchorState) => {
    switch (state) {
      case 'LOCKDOWN': return 'COGNITIVE LIMIT REACHED';
      case 'WARNING': return 'REALITY DRIFT DETECTED';
      case 'IDLE': return 'ANCHORED';
    }
  };

  // Calculate tether visualization points
  const tetherPoints = React.useMemo(() => {
    const points: { x: number; y: number; tension: number }[] = [];
    const maxPoints = Math.min(predictedReality.length, actualReality.length, 50);
    
    for (let i = 0; i < maxPoints; i++) {
      const pred = predictedReality[i];
      const actual = actualReality[i];
      const tension = Math.abs(pred - actual);
      
      points.push({
        x: (i / maxPoints) * 100,
        y: 50 + (pred - actual) * 20,
        tension,
      });
    }
    
    return points;
  }, [predictedReality, actualReality]);

  const avgTension = tetherPoints.reduce((sum, p) => sum + p.tension, 0) / tetherPoints.length || 0;

  return (
    <div className="relative p-4 bg-[#0a0a0c]/80 backdrop-blur-md rounded-lg border border-cyan-900/30">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div 
            className="w-3 h-3 rounded-full animate-pulse"
            style={{ backgroundColor: getStatusColor(realityAnchorState) }}
          />
          <span 
            className="text-sm font-mono font-bold tracking-wider"
            style={{ color: getStatusColor(realityAnchorState) }}
          >
            {getStatusText(realityAnchorState)}
          </span>
        </div>
        <div className="text-xs font-mono text-gray-500">
          KL-DIVERGENCE: {klDivergence.toFixed(4)}
        </div>
      </div>

      {/* Tether Visualization */}
      <div className="relative h-[150px] w-full overflow-hidden">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          {/* Background grid */}
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#1a1a2e" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#grid)" />

          {/* Center line (zero divergence) */}
          <line x1="0" y1="50" x2="100" y2="50" stroke="#2d3748" strokeWidth="0.5" strokeDasharray="2,2" />

          {/* Tether path */}
          {tetherPoints.length > 1 && (
            <path
              d={tetherPoints.reduce((acc, point, i) => {
                if (i === 0) return `M ${point.x} ${point.y}`;
                return `${acc} L ${point.x} ${point.y}`;
              }, '')}
              fill="none"
              stroke={avgTension > 0.3 ? '#ef4444' : avgTension > 0.1 ? '#fbbf24' : '#00ffff'}
              strokeWidth="1.5"
              className="transition-all duration-300"
            />
          )}

          {/* Tension indicators */}
          {tetherPoints.map((point, i) => (
            <circle
              key={i}
              cx={point.x}
              cy={point.y}
              r={Math.min(point.tension * 3, 2)}
              fill={point.tension > 0.3 ? '#ef4444' : point.tension > 0.1 ? '#fbbf24' : '#00ffff'}
              opacity={0.6}
            />
          ))}
        </svg>

        {/* Labels */}
        <div className="absolute left-2 top-2 text-[8px] font-mono text-cyan-600">PREDICTED</div>
        <div className="absolute left-2 bottom-2 text-[8px] font-mono text-magenta-600">ACTUAL</div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-2 mt-4">
        <MetricBox label="Avg Tension" value={avgTension.toFixed(3)} />
        <MetricBox label="Max Divergence" value={Math.max(...tetherPoints.map(p => p.tension), 0).toFixed(3)} />
        <MetricBox label="Samples" value={tetherPoints.length.toString()} />
      </div>
    </div>
  );
}

interface MetricBoxProps {
  label: string;
  value: string;
}

function MetricBox({ label, value }: MetricBoxProps) {
  return (
    <div className="p-2 bg-[#1a1a2e]/50 rounded border border-cyan-900/20 text-center">
      <div className="text-[10px] text-gray-500 font-mono uppercase">{label}</div>
      <div className="text-sm font-bold font-mono text-cyan-400">{value}</div>
    </div>
  );
}
