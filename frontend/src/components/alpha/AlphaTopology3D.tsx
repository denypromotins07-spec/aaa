'use client';

import React, { useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { AlphaNode, Vector3, updateAlphaTopology, mapConvictionToColor } from '../../utils/math/correlationToSpatial';
import { createAlphaNodeMaterial, updateAlphaNodeMaterial } from './AlphaNodeMaterial';

interface AlphaTopology3DProps {
  alphas: AlphaNode[];
  isRegimeShift: boolean;
}

function TopologyScene({ alphas, isRegimeShift }: AlphaTopology3DProps) {
  const groupRef = useRef<THREE.Group>(null);
  const nodeRefs = useRef<Map<string, { mesh: THREE.Mesh; material: THREE.ShaderMaterial }>>(new Map());
  const lineRefs = useRef<THREE.LineSegments | null>(null);
  const { size } = useThree();

  // Initialize nodes and lines
  useEffect(() => {
    if (!groupRef.current) return;

    const group = groupRef.current;
    
    // Clean up existing meshes
    group.children.forEach((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (child.material instanceof THREE.Material) {
          child.material.dispose();
        }
        group.remove(child);
      }
    });
    nodeRefs.current.clear();

    // Create new nodes
    alphas.forEach((alpha) => {
      const geometry = new THREE.SphereGeometry(1.5, 32, 32);
      const color = mapConvictionToColor(alpha.conviction);
      const material = createAlphaNodeMaterial(alpha.conviction, color);
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(alpha.position.x, alpha.position.y, alpha.position.z);
      mesh.userData.id = alpha.id;
      
      group.add(mesh);
      nodeRefs.current.set(alpha.id, { mesh, material });
    });

    // Create connection lines (for correlated pairs)
    const linePositions: number[] = [];
    alphas.forEach((a1, i) => {
      alphas.slice(i + 1).forEach((a2) => {
        // Draw line if in same cluster (simplified correlation visualization)
        if (Math.abs(a1.correlationCluster - a2.correlationCluster) < 0.5) {
          linePositions.push(a1.position.x, a1.position.y, a1.position.z);
          linePositions.push(a2.position.x, a2.position.y, a2.position.z);
        }
      });
    });

    if (linePositions.length > 0) {
      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
      const lineMaterial = new THREE.LineBasicMaterial({ 
        color: 0x00ffff, 
        transparent: true, 
        opacity: 0.3 
      });
      const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
      group.add(lines);
      lineRefs.current = lines;
    }

    return () => {
      // Cleanup on unmount or data change
      group.children.forEach((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (child.material instanceof THREE.Material) {
            child.material.dispose();
          }
        } else if (child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          if (child.material instanceof THREE.Material) {
            child.material.dispose();
          }
        }
      });
    };
  }, [alphas.length]); // Only recreate when node count changes

  // Animation loop
  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();

    // Update physics positions
    const updatedAlphas = updateAlphaTopology(alphas, isRegimeShift, { x: 0, y: 0, z: 0 });

    // Sync Three.js meshes with updated positions
    updatedAlphas.forEach((alpha) => {
      const nodeData = nodeRefs.current.get(alpha.id);
      if (nodeData) {
        const { mesh, material } = nodeData;
        
        // Smooth interpolation to new position
        mesh.position.lerp(new THREE.Vector3(alpha.position.x, alpha.position.y, alpha.position.z), 0.1);
        
        // Update shader uniforms
        const color = mapConvictionToColor(alpha.conviction);
        updateAlphaNodeMaterial(material, alpha.conviction, color);
        material.uniforms.uTime.value = time;
      }
    });

    // Rotate entire topology slowly
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05;
    }
  });

  return <group ref={groupRef} />;
}

export default function AlphaTopology3D({ alphas, isRegimeShift }: AlphaTopology3DProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [80, 60, 80], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#0a0a0c']} />
        <fog attach="fog" args={['#0a0a0c', 50, 200]} />
        <ambientLight intensity={0.5} />
        <pointLight position={[100, 100, 100]} intensity={1} color="#00ffff" />
        <pointLight position={[-100, -100, -100]} intensity={0.5} color="#ff00ff" />
        <TopologyScene alphas={alphas} isRegimeShift={isRegimeShift} />
      </Canvas>
      
      {/* Overlay Label */}
      <div className="absolute top-4 left-4 pointer-events-none">
        <h3 className="text-cyan-400 font-mono text-sm tracking-widest uppercase drop-shadow-[0_0_10px_rgba(0,255,255,0.8)]">
          Alpha Topology Manifold
        </h3>
        <p className="text-gray-400 font-mono text-xs mt-1">
          {isRegimeShift ? (
            <span className="text-red-400 animate-pulse">⚠ REGIME SHIFT DETECTED</span>
          ) : (
            'Orthogonal Strategy Clustering'
          )}
        </p>
      </div>
    </div>
  );
}
