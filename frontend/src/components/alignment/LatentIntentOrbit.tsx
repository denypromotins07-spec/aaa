'use client';

import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { ORBIT_VERTEX_SHADER, ORBIT_FRAGMENT_SHADER } from '../../utils/webgl/radarShaders';

interface IntentSphere {
  type: 'implicit' | 'explicit';
  position: [number, number, number];
  embeddings: number[];
}

interface LatentIntentOrbitProps {
  implicitSphere: IntentSphere;
  explicitSphere: IntentSphere;
  divergenceThreshold: number;
}

function Sphere({ position, color, radius = 0.15 }: { position: [number, number, number]; color: string; radius?: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const timeRef = useRef(0);
  const geometryRef = useRef<THREE.SphereGeometry | null>(null);
  const glowGeometryRef = useRef<THREE.SphereGeometry | null>(null);

  // Create geometries once
  useEffect(() => {
    geometryRef.current = new THREE.SphereGeometry(radius, 32, 32);
    glowGeometryRef.current = new THREE.SphereGeometry(radius * 1.4, 32, 32);
    
    if (meshRef.current) {
      meshRef.current.geometry = geometryRef.current;
    }
    if (glowRef.current) {
      glowRef.current.geometry = glowGeometryRef.current;
    }
    
    return () => {
      if (geometryRef.current) geometryRef.current.dispose();
      if (glowGeometryRef.current) glowGeometryRef.current.dispose();
    };
  }, [radius]);

  useFrame((state, delta) => {
    timeRef.current += delta;
    
    if (glowRef.current) {
      const scale = 1 + Math.sin(timeRef.current * 2) * 0.1;
      glowRef.current.scale.set(scale, scale, scale);
    }
  });

  // Cleanup materials
  useEffect(() => {
    return () => {
      if (meshRef.current && meshRef.current.material instanceof THREE.Material) {
        meshRef.current.material.dispose();
      }
      if (glowRef.current && glowRef.current.material instanceof THREE.Material) {
        glowRef.current.material.dispose();
      }
    };
  }, []);

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color}
          emissiveIntensity={0.5}
          transparent
          opacity={0.8}
        />
      </mesh>
      <mesh ref={glowRef}>
        <sphereGeometry args={[radius * 1.4, 32, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.2} />
      </mesh>
    </group>
  );
}

function Tether({ 
  start, 
  end, 
  isCritical 
}: { 
  start: [number, number, number]; 
  end: [number, number, number]; 
  isCritical: boolean;
}) {
  const lineRef = useRef<THREE.Line>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const timeRef = useRef(0);

  const points = useMemo(() => {
    const positions = new Float32Array([
      start[0], start[1], start[2],
      end[0], end[1], end[2],
    ]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Add color attribute for gradient
    const colors = new Float32Array([
      0, 1, 1, // cyan start
      isCritical ? 1 : 0, isCritical ? 0 : 1, isCritical ? 0.5, // end color
    ]);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    return geometry;
  }, [start, end, isCritical]);

  useFrame((state, delta) => {
    timeRef.current += delta;
    
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = timeRef.current;
      materialRef.current.uniforms.isCritical.value = isCritical ? 1.0 : 0.0;
    }
  });

  return (
    <line ref={lineRef} geometry={points}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={ORBIT_VERTEX_SHADER}
        fragmentShader={ORBIT_FRAGMENT_SHADER}
        transparent
        uniforms={{
          uTime: { value: 0 },
          isCritical: { value: isCritical ? 1.0 : 0.0 },
          uAlpha: { value: 0.8 },
        }}
      />
    </line>
  );
}

export function LatentIntentOrbit({ 
  implicitSphere, 
  explicitSphere, 
  divergenceThreshold 
}: LatentIntentOrbitProps) {
  const distance = Math.sqrt(
    Math.pow(implicitSphere.position[0] - explicitSphere.position[0], 2) +
    Math.pow(implicitSphere.position[1] - explicitSphere.position[1], 2) +
    Math.pow(implicitSphere.position[2] - explicitSphere.position[2], 2)
  );
  
  const isCritical = distance > divergenceThreshold;

  return (
    <div className="w-full h-[300px] relative">
      <Canvas camera={{ position: [0, 0, 3], fov: 45 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        
        <Sphere 
          position={implicitSphere.position} 
          color="#ff00ff" 
          radius={0.15}
        />
        <Sphere 
          position={explicitSphere.position} 
          color="#00ffff" 
          radius={0.15}
        />
        
        <Tether 
          start={implicitSphere.position} 
          end={explicitSphere.position} 
          isCritical={isCritical}
        />
      </Canvas>
      
      {isCritical && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-red-900/80 border border-red-500 rounded text-red-200 text-sm font-mono animate-pulse">
          ⚠️ INTENT DIVERGENCE DETECTED
        </div>
      )}
    </div>
  );
}
