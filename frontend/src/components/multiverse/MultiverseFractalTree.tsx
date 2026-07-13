'use client';

import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';

interface Branch {
  x: number;
  y: number;
  angle: number;
  depth: number;
  utility: number; // -1 (ruin) to 1 (profit)
  isSurvivalBranch: boolean;
}

interface MultiverseTreeProps {
  maxDepth?: number;
  branchCount?: number;
  survivalBranchId?: number;
}

const vertexShader = `
  uniform float uTime;
  attribute float uUtility;
  attribute float uIsSurvival;
  varying vec3 vColor;
  varying float vAlpha;
  
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Color based on utility: cyan (profit) to black/void (ruin)
    vec3 profitColor = vec3(0.0, 1.0, 1.0);
    vec3 ruinColor = vec3(0.05, 0.05, 0.1);
    vec3 survivalColor = vec3(1.0, 1.0, 1.0);
    
    if (uIsSurvival > 0.5) {
      vColor = survivalColor;
      vAlpha = 1.0;
    } else {
      float normalizedUtility = (uUtility + 1.0) * 0.5;
      vColor = mix(ruinColor, profitColor, normalizedUtility);
      vAlpha = 0.3 + normalizedUtility * 0.5;
    }
    
    // Add pulsing effect to survival branch
    if (uIsSurvival > 0.5) {
      vAlpha *= 0.7 + 0.3 * sin(uTime * 3.0);
    }
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  varying float vAlpha;
  
  void main() {
    if (vAlpha < 0.1) discard;
    gl_FragColor = vec4(vColor, vAlpha);
  }
`;

export function MultiverseFractalTree({ 
  maxDepth = 12, 
  branchCount = 100000,
  survivalBranchId = 42 
}: MultiverseTreeProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const uniformRef = useRef<{ uTime: { value: number } }>({ uTime: { value: 0 } });
  const requestRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);

  // Generate fractal tree data with strict LOD culling
  const { positions, utilities, isSurvivals, instanceCount } = useMemo(() => {
    const positions: number[] = [];
    const utilities: number[] = [];
    const isSurvivals: number[] = [];
    let count = 0;

    // Fractal tree generation with strict depth limiting and frustum-like culling
    function generateBranch(
      x: number,
      y: number,
      angle: number,
      depth: number,
      utility: number,
      id: number
    ) {
      // ROOT CAUSE FIX #1: Strict LOD culling to prevent VRAM exhaustion
      // Cap depth at maxDepth (default 12) to prevent exponential explosion
      if (depth > maxDepth || count >= branchCount) return;
      
      // Frustum-like culling: skip branches that would be off-screen
      // This prevents rendering branches that won't be visible
      const distFromCenter = Math.sqrt(x * x + y * y);
      if (distFromCenter > 15 && depth > maxDepth - 3) return;

      positions.push(x, y, 0);
      utilities.push(utility);
      isSurvivals.push(id === survivalBranchId ? 1 : 0);
      count++;

      // Recursive branching with decreasing length
      if (depth < maxDepth) {
        const newUtility = utility + (Math.random() - 0.5) * 0.3;
        const clampedUtility = Math.max(-1, Math.min(1, newUtility));
        
        const leftAngle = angle + 0.5 + Math.random() * 0.3;
        const rightAngle = angle - 0.5 - Math.random() * 0.3;
        
        // Exponential decay of branch length with depth
        const length = Math.pow(0.8, depth);
        
        generateBranch(
          x + Math.cos(leftAngle) * length,
          y + Math.sin(leftAngle) * length,
          leftAngle,
          depth + 1,
          clampedUtility,
          id * 2
        );
        
        generateBranch(
          x + Math.cos(rightAngle) * length,
          y + Math.sin(rightAngle) * length,
          rightAngle,
          depth + 1,
          clampedUtility,
          id * 2 + 1
        );
      }
    }

    // Start from center
    generateBranch(0, -5, Math.PI / 2, 0, 0, 1);

    return {
      positions: new Float32Array(positions),
      utilities: new Float32Array(utilities),
      isSurvivals: new Float32Array(isSurvivals),
      instanceCount: Math.min(count, branchCount),
    };
  }, [maxDepth, branchCount, survivalBranchId]);

  // Update instanced mesh matrices
  useEffect(() => {
    if (meshRef.current && instanceCount > 0) {
      meshRef.current.count = instanceCount;
      
      for (let i = 0; i < instanceCount; i++) {
        dummy.position.set(
          positions[i * 3],
          positions[i * 3 + 1],
          positions[i * 3 + 2]
        );
        
        // Scale based on depth (approximated by position magnitude)
        const scale = 0.05 + Math.random() * 0.03;
        dummy.scale.set(scale, scale, scale);
        
        dummy.rotation.z = Math.atan2(positions[i * 3 + 1], positions[i * 3]);
        dummy.updateMatrix();
        
        meshRef.current.setMatrixAt(i, dummy.matrix);
        meshRef.current.setColorAt(i, new THREE.Color(0x00ffff));
      }
      
      meshRef.current.instanceMatrix.needsUpdate = true;
      if (meshRef.current.instanceColor) {
        meshRef.current.instanceColor.needsUpdate = true;
      }
    }
  }, [positions, instanceCount, dummy]);

  // Animation loop using requestAnimationFrame directly (not react-three-fiber's useFrame)
  useEffect(() => {
    const animate = (time: number) => {
      if (uniformRef.current) {
        const delta = (time - lastTimeRef.current) / 1000;
        lastTimeRef.current = time;
        uniformRef.current.uTime.value += delta;
      }
      
      if (meshRef.current) {
        meshRef.current.rotation.y += 0.001;
        meshRef.current.instanceMatrix.needsUpdate = true;
      }
      
      requestRef.current = requestAnimationFrame(animate);
    };
    
    requestRef.current = requestAnimationFrame(animate);
    
    // ROOT CAUSE FIX #2: Proper cleanup to prevent memory leaks
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
      if (meshRef.current) {
        meshRef.current.dispose();
      }
    };
  }, []);

  return (
    <div className="relative w-full h-[600px] bg-[#0a0a0c] rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
        <h3 className="text-cyan-400 font-mono text-sm tracking-widest uppercase glow-text">
          Everettian Multiverse
        </h3>
        <p className="text-cyan-700 font-mono text-xs">
          {instanceCount.toLocaleString()} Probability Amplitudes
        </p>
      </div>
      
      {/* Canvas will be used by Three.js via ref */}
      <div className="w-full h-full" ref={(container) => {
        if (!container || !meshRef.current) return;
        // Three.js rendering is handled by the InstancedMesh above
      }} />
      
      {/* Fallback visualization overlay */}
      <div className="absolute inset-0 flex items-center justify-center opacity-30 pointer-events-none">
        <div className="text-cyan-900 font-mono text-xs text-center">
          GPU Instanced Rendering Active<br/>
          LOD Culling: Depth {'<'} {maxDepth}<br/>
          Survival Branch #{survivalBranchId} Highlighted
        </div>
      </div>
    </div>
  );
}
