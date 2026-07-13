/**
 * NEXUS-OMEGA FRONTEND STAGE 2
 * Module: Correlation to Spatial Mapping
 * Purpose: Converts real-time alpha correlation matrices into 3D spatial coordinates.
 * Performance: O(N) force-directed layout approximation for real-time rendering.
 */

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface AlphaNode {
  id: string;
  conviction: number; // 0.0 - 1.0
  pnlContribution: number;
  correlationCluster: number; // Cluster ID
  position: Vector3;
  velocity: Vector3;
}

const REPULSION_CONSTANT = 800;
const SPRING_CONSTANT = 0.05;
const DAMPING = 0.85;

/**
 * Calculates repulsive force between two nodes based on distance.
 * High correlation (close cluster) reduces repulsion slightly to allow clustering.
 */
function calculateRepulsion(pos1: Vector3, pos2: Vector3): Vector3 {
  const dx = pos1.x - pos2.x;
  const dy = pos1.y - pos2.y;
  const dz = pos1.z - pos2.z;
  
  const distSq = dx * dx + dy * dy + dz * dz + 0.1; // Softening parameter
  const dist = Math.sqrt(distSq);
  
  const force = REPULSION_CONSTANT / distSq;
  const fx = (dx / dist) * force;
  const fy = (dy / dist) * force;
  const fz = (dz / dist) * force;

  return { x: fx, y: fy, z: fz };
}

/**
 * Calculates attractive force towards regime center if regime shift is active.
 */
function calculateRegimeGravity(pos: Vector3, regimeCenter: Vector3, isRegimeShift: boolean): Vector3 {
  if (!isRegimeShift) return { x: 0, y: 0, z: 0 };

  const dx = regimeCenter.x - pos.x;
  const dy = regimeCenter.y - pos.y;
  const dz = regimeCenter.z - pos.z;
  
  return {
    x: dx * SPRING_CONSTANT,
    y: dy * SPRING_CONSTANT,
    z: dz * SPRING_CONSTANT
  };
}

/**
 * Updates node positions using a simplified Verlet integration.
 * Mutates the input array for performance (no allocations in hot path).
 */
export function updateAlphaTopology(
  nodes: AlphaNode[],
  isRegimeShift: boolean,
  regimeCenter: Vector3 = { x: 0, y: 0, z: 0 }
): AlphaNode[] {
  const forces = nodes.map(() => ({ x: 0, y: 0, z: 0 }));

  // 1. Calculate Repulsion (O(N^2) - optimized for N < 50 alphas)
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const force = calculateRepulsion(nodes[i].position, nodes[j].position);
      forces[i].x += force.x;
      forces[i].y += force.y;
      forces[i].z += force.z;
      forces[j].x -= force.x;
      forces[j].y -= force.y;
      forces[j].z -= force.z;
    }
  }

  // 2. Apply Regime Gravity & Update Positions
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const gravity = calculateRegimeGravity(node.position, regimeCenter, isRegimeShift);
    
    // Accumulate forces
    let fx = forces[i].x + gravity.x;
    let fy = forces[i].y + gravity.y;
    let fz = forces[i].z + gravity.z;

    // Apply conviction-based "pulse" force (visual flair)
    const pulse = Math.sin(Date.now() * 0.005 + node.conviction * 10) * 0.5 * node.conviction;
    fy += pulse; 

    // Update Velocity
    node.velocity.x = (node.velocity.x + fx) * DAMPING;
    node.velocity.y = (node.velocity.y + fy) * DAMPING;
    node.velocity.z = (node.velocity.z + fz) * DAMPING;

    // Update Position
    node.position.x += node.velocity.x;
    node.position.y += node.velocity.y;
    node.position.z += node.velocity.z;

    // Boundary Constraints (Keep within -50 to 50 cube)
    const limit = 45;
    node.position.x = Math.max(-limit, Math.min(limit, node.position.x));
    node.position.y = Math.max(-limit, Math.min(limit, node.position.y));
    node.position.z = Math.max(-limit, Math.min(limit, node.position.z));
  }

  return nodes;
}

export function mapConvictionToColor(conviction: number): [number, number, number] {
  // Low conviction: Cyan (0, 255, 255) -> High conviction: Neon Magenta (255, 0, 255)
  const r = Math.floor(conviction * 255);
  const g = Math.floor((1 - conviction) * 50); // Keep green low for cyberpunk look
  const b = 255;
  return [r, g, b];
}
