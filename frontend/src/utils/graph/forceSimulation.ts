import { SimulationLinkDatum, SimulationNodeDatum } from 'd3-force';

export interface SwarmNode extends SimulationNodeDatum {
  id: string;
  role: 'leader' | 'follower' | 'candidate' | 'fenced';
  latency: number; // ms
  health: number; // 0-1
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  shatterProgress?: number; // For STONITH animation
}

export interface SwarmLink extends SimulationLinkDatum<SwarmNode> {
  source: string | SwarmNode;
  target: string | SwarmNode;
  latency: number;
  bandwidth: number;
}

// Force simulation parameters optimized for performance
export const FORCE_SIMULATION_CONFIG = {
  charge: -200, // Repulsion between nodes
  linkDistance: 150, // Ideal link length
  linkStrength: 0.1, // Link stiffness
  collisionRadius: 30, // Node collision radius
  centerGravity: 0.05, // Pull towards center
  decay: 0.98, // Velocity decay per tick
};

// Convert latency to edge thickness and color
export function latencyToEdgeStyle(latency: number): { width: number; color: string } {
  if (latency < 10) {
    return { width: 1, color: '#06b6d4' }; // Cyan - excellent
  } else if (latency < 50) {
    return { width: 2, color: '#22c55e' }; // Green - good
  } else if (latency < 100) {
    return { width: 3, color: '#eab308' }; // Yellow - moderate
  } else if (latency < 200) {
    return { width: 4, color: '#f97316' }; // Orange - high
  } else {
    return { width: 6, color: '#ef4444' }; // Red - critical
  }
}

// Calculate node color based on role and health
export function getNodeColor(node: SwarmNode): string {
  if (node.role === 'fenced') {
    return '#ef4444'; // Red for fenced nodes
  }
  if (node.role === 'leader') {
    return '#fbbf24'; // Gold for leader
  }
  if (node.role === 'candidate') {
    return '#a855f7'; // Purple for candidate
  }
  // Follower color based on health
  const intensity = Math.floor(node.health * 255);
  return `rgb(6, ${intensity}, ${intensity})`;
}

// STONITH shatter effect calculation
export function calculateShatterEffect(
  node: SwarmNode,
  explosionCenter: { x: number; y: number },
  time: number
): { x: number; y: number; opacity: number } {
  if (node.role !== 'fenced' || node.shatterProgress === undefined) {
    return { x: node.x || 0, y: node.y || 0, opacity: 1 };
  }

  const dx = (node.x || 0) - explosionCenter.x;
  const dy = (node.y || 0) - explosionCenter.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx);

  // Shatter progress: 0 to 1
  const progress = Math.min(node.shatterProgress, 1);
  
  // Particles fly outward exponentially
  const velocity = 50 * progress * progress;
  const scatterX = dx / distance * velocity + (Math.random() - 0.5) * progress * 20;
  const scatterY = dy / distance * velocity + (Math.random() - 0.5) * progress * 20;

  // Fade out as shatter completes
  const opacity = 1 - progress;

  return {
    x: (node.x || 0) + scatterX,
    y: (node.y || 0) + scatterY,
    opacity,
  };
}

// Throttle function for force simulation ticks
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  let lastResult: ReturnType<T>;

  return function (this: any, ...args: Parameters<T>): ReturnType<T> {
    if (!inThrottle) {
      inThrottle = true;
      lastResult = func.apply(this, args);
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
    return lastResult;
  };
}
