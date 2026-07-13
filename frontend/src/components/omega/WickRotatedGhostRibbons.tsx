'use client';

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface GhostRibbonsProps {
  alphaExhaustion: number;
}

const ghostVertexShader = `
  uniform float uTime;
  uniform float uAlphaExhaustion;
  attribute float aIndex;
  varying vec3 vColor;
  varying float vOpacity;
  
  void main() {
    float t = uTime * 0.15;
    float idx = aIndex * 0.02;
    
    // Wick-rotated imaginary time simulation
    float x = sin(t * 0.7 + idx) * cos(idx * 0.3);
    float y = cos(t * 0.5 + idx * 0.8) * sin(idx * 0.2);
    float z = sin(t * 0.3 + idx * 0.5) * 0.5;
    
    // Fade based on alpha exhaustion (ghost ribbons appear more as alpha depletes)
    float appearance = uAlphaExhaustion * 0.6;
    
    vec3 newPos = position;
    newPos.x += x * 3.0;
    newPos.y += y * 3.0;
    newPos.z += z * 3.0 - 2.0; // Offset behind main attractor
    
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Ethereal cyan-blue gradient
    vColor = vec3(0.2, 0.6, 0.9);
    vOpacity = appearance * (0.3 + 0.2 * sin(t + idx));
  }
`;

const ghostFragmentShader = `
  varying vec3 vColor;
  varying float vOpacity;
  
  void main() {
    if (vOpacity < 0.05) discard;
    gl_FragColor = vec4(vColor, vOpacity * 0.5);
  }
`;

export function WickRotatedGhostRibbons({ alphaExhaustion }: GhostRibbonsProps) {
  const meshRef = useRef<THREE.Points>(null);
  const uniformRef = useRef<{ uTime: { value: number }; uAlphaExhaustion: { value: number } }>({
    uTime: { value: 0 },
    uAlphaExhaustion: { value: alphaExhaustion },
  });

  const ribbonCount = 20000;

  const geometry = React.useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(ribbonCount * 3);
    const indices = new Float32Array(ribbonCount);

    for (let i = 0; i < ribbonCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 0.1;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 0.1;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 0.1;
      indices[i] = i;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aIndex', new THREE.BufferAttribute(indices, 1));
    return geo;
  }, []);

  useFrame((state, delta) => {
    if (uniformRef.current) {
      uniformRef.current.uTime.value += delta;
      uniformRef.current.uAlphaExhaustion.value = alphaExhaustion;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y -= delta * 0.05;
    }
  });

  return (
    <points ref={meshRef} geometry={geometry}>
      <shaderMaterial
        vertexShader={ghostVertexShader}
        fragmentShader={ghostFragmentShader}
        uniforms={uniformRef.current}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        side={THREE.DoubleSide}
      />
    </points>
  );
}
