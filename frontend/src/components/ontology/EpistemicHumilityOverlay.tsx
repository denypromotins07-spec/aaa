'use client';

import React from 'react';
import { useRealityAnchor, useIsEpistemicLockdown } from '../../hooks/useRealityAnchor';

interface EpistemicHumilityOverlayProps {
  onRecalibrate?: () => void;
}

export function EpistemicHumilityOverlay({ onRecalibrate }: EpistemicHumilityOverlayProps) {
  const realityAnchorState = useRealityAnchor(state => state.realityAnchorState);
  const isEpistemicLockdown = useIsEpistemicLockdown();
  const acknowledgeRecalibration = useRealityAnchor(state => state.acknowledgeRecalibration);

  // Only render overlay in WARNING or LOCKDOWN states
  if (realityAnchorState === 'IDLE') {
    return null;
  }

  const handleRecalibrate = () => {
    acknowledgeRecalibration();
    onRecalibrate?.();
  };

  return (
    <>
      {/* Full-screen crimson overlay */}
      <div 
        className="fixed inset-0 z-[9999] pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at center, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.3) 50%, rgba(69, 10, 10, 0.5) 100%)',
          backdropFilter: 'blur(4px)',
        }}
      >
        {/* Animated border pulse */}
        <div 
          className="absolute inset-0 border-4 border-red-600/50 rounded-lg animate-pulse"
          style={{ animationDuration: '2s' }}
        />

        {/* Central warning message */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div 
            className="bg-[#0a0a0c]/95 backdrop-blur-xl border-2 border-red-500 rounded-xl p-8 max-w-lg mx-4 shadow-2xl shadow-red-900/50"
            style={{
              boxShadow: '0 0 60px rgba(239, 68, 68, 0.4), inset 0 0 30px rgba(239, 68, 68, 0.1)',
            }}
          >
            {/* Warning icon */}
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 rounded-full bg-red-900/30 border-2 border-red-500 flex items-center justify-center animate-pulse">
                <svg className="w-10 h-10 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
            </div>

            {/* Main warning text */}
            <h1 className="text-2xl font-bold text-red-500 text-center font-mono tracking-wider mb-2">
              COGNITIVE LIMIT REACHED
            </h1>
            
            <p className="text-red-400/80 text-center text-sm font-mono mb-6">
              ONTOLOGICAL CRISIS DETECTED - EPISTEMIC HUMILITY MODE ENGAGED
            </p>

            {/* Status indicators */}
            <div className="grid grid-cols-2 gap-3 mb-6">
              <StatusItem label="KL-Divergence" value="CRITICAL" color="#ef4444" />
              <StatusItem label="Reality Anchor" value="UNSTABLE" color="#fbbf24" />
              <StatusItem label="Trading Status" value="SUSPENDED" color="#ef4444" />
              <StatusItem label="Control Plane" value="LOCKED" color="#ef4444" />
            </div>

            {/* Explanation */}
            <div className="bg-red-900/20 border border-red-800/50 rounded p-3 mb-6">
              <p className="text-xs text-red-300 font-mono leading-relaxed">
                The AI&apos;s internal world-model has detected a structural break in market dynamics 
                that exceeds its comprehension threshold. All execution controls have been disabled 
                until reality recalibration is complete.
              </p>
            </div>

            {/* Recalibration button (only enabled when safe) */}
            {realityAnchorState === 'WARNING' && (
              <button
                onClick={handleRecalibrate}
                className="w-full py-3 px-4 bg-red-900/50 hover:bg-red-800/50 border border-red-600 rounded-lg text-red-200 font-mono text-sm transition-all duration-200 hover:shadow-lg hover:shadow-red-900/30"
              >
                ACKNOWLEDGE & RECALIBRATE
              </button>
            )}
            
            {realityAnchorState === 'LOCKDOWN' && (
              <div className="w-full py-3 px-4 bg-red-950/50 border border-red-700 rounded-lg text-red-400 font-mono text-sm text-center animate-pulse">
                ⚠️ AWAITING KL-DIVERGENCE REDUCTION...
              </div>
            )}
          </div>
        </div>

        {/* Corner warnings */}
        <div className="absolute top-4 left-4 px-3 py-2 bg-red-900/40 backdrop-blur border border-red-600/50 rounded text-red-400 font-mono text-xs">
          ⚠️ EPISTEMIC HUMILITY ACTIVE
        </div>
        <div className="absolute top-4 right-4 px-3 py-2 bg-red-900/40 backdrop-blur border border-red-600/50 rounded text-red-400 font-mono text-xs animate-pulse">
          ALL CONTROLS DISABLED
        </div>
        <div className="absolute bottom-4 left-4 px-3 py-2 bg-red-900/40 backdrop-blur border border-red-600/50 rounded text-red-400 font-mono text-xs">
          DO NOT ATTEMPT MANUAL OVERRIDE
        </div>
        <div className="absolute bottom-4 right-4 px-3 py-2 bg-red-900/40 backdrop-blur border border-red-600/50 rounded text-red-400 font-mono text-xs">
          SYSTEM PROTOCOL: STAGE-24/33
        </div>
      </div>

      {/* Global CSS class to apply grayscale and disable pointer events on all interactive elements */}
      <style jsx global>{`
        ${isEpistemicLockdown ? `
          body {
            filter: grayscale(100%);
          }
          
          /* Disable all interactive elements */
          button, input, select, textarea, [role="button"], [tabindex] {
            pointer-events: none !important;
            opacity: 0.5 !important;
            cursor: not-allowed !important;
          }
          
          /* Exception: the overlay itself */
          .z-\\[9999\\], .z-\\[9999\\] * {
            pointer-events: auto !important;
            filter: none !important;
          }
        ` : ''}
      `}</style>
    </>
  );
}

interface StatusItemProps {
  label: string;
  value: string;
  color: string;
}

function StatusItem({ label, value, color }: StatusItemProps) {
  return (
    <div className="bg-[#0a0a0c]/50 border border-gray-800 rounded p-2">
      <div className="text-[10px] text-gray-500 font-mono uppercase">{label}</div>
      <div className="text-sm font-bold font-mono" style={{ color }}>
        {value}
      </div>
    </div>
  );
}
