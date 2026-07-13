'use client';

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface LandauerGaugeProps {
  value: number; // 0 to 1, where 1 = maximum alpha exhaustion
}

const sphereVertexShader = `
  uniform float uCollapseFactor;
  varying vec3 vNormal;
  varying vec3 vPosition;
  
  void main() {
    vNormal = normal;
    vPosition = position;
    
    // Collapse sphere based on alpha exhaustion
    vec3 newPos = position * (1.0 - uCollapseFactor * 0.8);
    
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const sphereFragmentShader = `
  uniform float uCollapseFactor;
  varying vec3 vNormal;
  varying vec3 vPosition;
  
  void main() {
    // Holographic grid pattern
    float grid = sin(vPosition.x * 20.0) * sin(vPosition.y * 20.0) * sin(vPosition.z * 20.0);
    float hologram = smoothstep(0.0, 0.1, abs(grid));
    
    // Color shifts from cyan (healthy) to red (exhausted)
    vec3 healthyColor = vec3(0.0, 1.0, 1.0);
    vec3 exhaustedColor = vec3(1.0, 0.2, 0.2);
    vec3 color = mix(healthyColor, exhaustedColor, uCollapseFactor);
    
    float alpha = 0.3 + 0.4 * hologram;
    gl_FragColor = vec4(color, alpha);
  }
`;

export function LandauerThermodynamicGauge({ value }: LandauerGaugeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const uniformRef = useRef<{ uCollapseFactor: { value: number } }>({
    uCollapseFactor: { value: value },
  });

  useFrame(() => {
    if (uniformRef.current && meshRef.current) {
      uniformRef.current.uCollapseFactor.value = value;
      meshRef.current.rotation.y += 0.01;
      meshRef.current.rotation.z += 0.005;
    }
  });

  return (
    <div className="relative w-32 h-32">
      <div className="absolute inset-0 flex items-center justify-center">
        <canvas className="w-full h-full" />
      </div>
      
      {/* Simple CSS-based gauge fallback/overlay */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <div 
          className="w-20 h-20 rounded-full border-2 transition-all duration-500"
          style={{
            borderColor: value > 0.8 ? '#ef4444' : value > 0.5 ? '#f59e0b' : '#06b6d4',
            boxShadow: `0 0 ${20 + value * 30}px ${value > 0.8 ? '#ef4444' : value > 0.5 ? '#f59e0b' : '#06b6d4'}`,
            transform: `scale(${1 - value * 0.6})`,
          }}
        />
        <span className="mt-2 text-xs font-mono" style={{ color: value > 0.8 ? '#ef4444' : value > 0.5 ? '#f59e0b' : '#06b6d4' }}>
          {(value * 100).toFixed(1)}% EXHAUSTED
        </span>
      </div>
      
      {/* Three.js visualization (hidden for now, can be enabled) */}
      <div className="hidden">
        {/* 
        <Canvas camera={{ position: [0, 0, 3], fov: 50 }}>
          <mesh ref={meshRef}>
            <sphereGeometry args={[1, 32, 32]} />
            <shaderMaterial
              vertexShader={sphereVertexShader}
              fragmentShader={sphereFragmentShader}
              uniforms={uniformRef.current}
              transparent
              side={THREE.DoubleSide}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        </Canvas>
        */}
      </div>
    </div>
  );
}
