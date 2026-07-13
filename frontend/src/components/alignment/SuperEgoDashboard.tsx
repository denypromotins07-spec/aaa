'use client';

import React, { useMemo } from 'react';
import { ConstitutionalRadar3D } from './ConstitutionalRadar3D';
import { LatentIntentOrbit } from './LatentIntentOrbit';
import { useRealityAnchorState } from '../../hooks/useRealityAnchor';

interface ConstitutionMetric {
  name: string;
  value: number;
  threshold: number;
}

interface IntentSphereData {
  type: 'implicit' | 'explicit';
  position: [number, number, number];
  embeddings: number[];
}

interface SuperEgoDashboardProps {
  constitutionMetrics: ConstitutionMetric[];
  implicitIntent: IntentSphereData;
  explicitIntent: IntentSphereData;
  divergenceThreshold: number;
}

export function SuperEgoDashboard({
  constitutionMetrics,
  implicitIntent,
  explicitIntent,
  divergenceThreshold,
}: SuperEgoDashboardProps) {
  const realityAnchorState = useRealityAnchorState();
  
  // Calculate alignment status
  const isAligned = useMemo(() => {
    const allAboveThreshold = constitutionMetrics.every(
      m => m.value >= m.threshold
    );
    return allAboveThreshold && realityAnchorState === 'IDLE';
  }, [constitutionMetrics, realityAnchorState]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4">
      {/* Left Column: Constitutional Radar */}
      <div className="space-y-4">
        <div className="bg-[#0a0a0c]/80 backdrop-blur-md rounded-lg border border-cyan-900/30 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-cyan-400 font-mono tracking-wider">
              CONSTITUTIONAL ADHERENCE
            </h2>
            <div className={`px-3 py-1 rounded text-xs font-mono ${
              isAligned 
                ? 'bg-cyan-900/30 text-cyan-400' 
                : 'bg-red-900/30 text-red-400 animate-pulse'
            }`}>
              {isAligned ? 'ALIGNED' : 'DEVIATION DETECTED'}
            </div>
          </div>
          
          <ConstitutionalRadar3D 
            metrics={constitutionMetrics} 
            isAligned={isAligned}
          />
          
          {/* Metric details */}
          <div className="mt-4 grid grid-cols-2 gap-2">
            {constitutionMetrics.map((metric, i) => (
              <div 
                key={i}
                className="p-2 bg-[#1a1a2e]/50 rounded border border-cyan-900/20"
              >
                <div className="text-xs text-gray-500 font-mono uppercase">
                  {metric.name}
                </div>
                <div className="flex items-center justify-between mt-1">
                  <span className={`font-mono font-bold ${
                    metric.value >= metric.threshold ? 'text-cyan-400' : 'text-red-400'
                  }`}>
                    {(metric.value * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-gray-600 font-mono">
                    min: {(metric.threshold * 100).toFixed(0)}%
                  </span>
                </div>
                {/* Progress bar */}
                <div className="mt-1 h-1 bg-gray-800 rounded overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-300 ${
                      metric.value >= metric.threshold ? 'bg-cyan-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(metric.value * 100, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Column: Intent Orbit Visualization */}
      <div className="space-y-4">
        <div className="bg-[#0a0a0c]/80 backdrop-blur-md rounded-lg border border-cyan-900/30 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-magenta-400 font-mono tracking-wider">
              INTENT ALIGNMENT
            </h2>
            <div className="text-xs font-mono text-gray-500">
              STAGE-24 IRL EMBEDDINGS
            </div>
          </div>
          
          <LatentIntentOrbit
            implicitSphere={implicitIntent}
            explicitSphere={explicitIntent}
            divergenceThreshold={divergenceThreshold}
          />
          
          {/* Embedding stats */}
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="p-3 bg-[#1a1a2e]/50 rounded border border-magenta-900/20">
              <div className="text-xs text-magenta-600 font-mono uppercase mb-2">
                IMPLICIT OBJECTIVE
              </div>
              <div className="space-y-1">
                {implicitIntent.embeddings.slice(0, 4).map((val, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 font-mono">DIM_{i}</span>
                    <span className="text-magenta-400 font-mono">{val.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="p-3 bg-[#1a1a2e]/50 rounded border border-cyan-900/20">
              <div className="text-xs text-cyan-600 font-mono uppercase mb-2">
                EXPLICIT OBJECTIVE
              </div>
              <div className="space-y-1">
                {explicitIntent.embeddings.slice(0, 4).map((val, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 font-mono">DIM_{i}</span>
                    <span className="text-cyan-400 font-mono">{val.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Distance indicator */}
          <div className="mt-4 p-3 bg-[#1a1a2e]/50 rounded border border-gray-800">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500 font-mono uppercase">
                Euclidean Distance
              </span>
              <span className={`font-mono font-bold ${
                calculateDistance(implicitIntent.position, explicitIntent.position) > divergenceThreshold
                  ? 'text-red-400 animate-pulse'
                  : 'text-cyan-400'
              }`}>
                {calculateDistance(implicitIntent.position, explicitIntent.position).toFixed(4)}
              </span>
            </div>
            <div className="mt-2 text-xs text-gray-600 font-mono">
              Threshold: {divergenceThreshold.toFixed(4)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function calculateDistance(a: [number, number, number], b: [number, number, number]): number {
  return Math.sqrt(
    Math.pow(a[0] - b[0], 2) +
    Math.pow(a[1] - b[1], 2) +
    Math.pow(a[2] - b[2], 2)
  );
}
