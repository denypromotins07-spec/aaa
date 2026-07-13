'use client';

import { create } from 'zustand';

export type RealityAnchorState = 'IDLE' | 'WARNING' | 'LOCKDOWN';

interface RealityAnchorStore {
  // State
  klDivergence: number;
  realityAnchorState: RealityAnchorState;
  isEpistemicLockdown: boolean;
  lastDriftEvent: number | null;
  
  // Actions
  updateKLDivergence: (value: number) => void;
  triggerWarning: () => void;
  triggerLockdown: () => void;
  resetToIdle: () => void;
  acknowledgeRecalibration: () => void;
}

const KL_WARNING_THRESHOLD = 0.7;
const KL_LOCKDOWN_THRESHOLD = 0.9;

export const useRealityAnchor = create<RealityAnchorStore>((set, get) => ({
  klDivergence: 0,
  realityAnchorState: 'IDLE',
  isEpistemicLockdown: false,
  lastDriftEvent: null,

  updateKLDivergence: (value: number) => {
    const currentState = get().realityAnchorState;
    
    set({ 
      klDivergence: value,
      lastDriftEvent: Date.now()
    });

    // State machine transitions
    if (value >= KL_LOCKDOWN_THRESHOLD && currentState !== 'LOCKDOWN') {
      get().triggerLockdown();
    } else if (value >= KL_WARNING_THRESHOLD && value < KL_LOCKDOWN_THRESHOLD && currentState === 'IDLE') {
      get().triggerWarning();
    } else if (value < KL_WARNING_THRESHOLD && currentState !== 'IDLE') {
      get().resetToIdle();
    }
  },

  triggerWarning: () => {
    set({ 
      realityAnchorState: 'WARNING',
      isEpistemicLockdown: false 
    });
    console.warn('[REALITY ANCHOR] Warning state triggered - KL divergence elevated');
  },

  triggerLockdown: () => {
    set({ 
      realityAnchorState: 'LOCKDOWN',
      isEpistemicLockdown: true 
    });
    console.error('[REALITY ANCHOR] LOCKDOWN TRIGGERED - Epistemic humility mode engaged');
  },

  resetToIdle: () => {
    set({ 
      realityAnchorState: 'IDLE',
      isEpistemicLockdown: false 
    });
  },

  acknowledgeRecalibration: () => {
    const { klDivergence } = get();
    
    // Only allow reset if KL has returned to safe levels
    if (klDivergence < KL_WARNING_THRESHOLD) {
      set({ 
        realityAnchorState: 'IDLE',
        isEpistemicLockdown: false 
      });
      console.log('[REALITY ANCHOR] Recalibration acknowledged - returning to idle');
    } else {
      console.warn('[REALITY ANCHOR] Cannot recalibrate - KL divergence still elevated');
    }
  },
}));

// Hook for components to check lockdown status
export const useIsEpistemicLockdown = () => {
  return useRealityAnchor(state => state.isEpistemicLockdown);
};

// Hook to get current state
export const useRealityAnchorState = () => {
  return useRealityAnchor(state => state.realityAnchorState);
};
