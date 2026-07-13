'use client';

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { WickRotatedGhostRibbons } from './WickRotatedGhostRibbons';
import { LandauerThermodynamicGauge } from './LandauerThermodynamicGauge';

// Shader for the strange attractor ribbon
const attractorVertexShader = `
  uniform float uTime;
  uniform float uAlphaExhaustion;
  attribute float aIndex;
  varying vec3 vColor;
  varying float vAlpha;
  
  // Simplex noise function (simplified for brevity)
  vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
  float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    float t = uTime * 0.2;
    float idx = aIndex * 0.01;
    
    // Takens embedding simulation
    float x = sin(t + idx) + snoise(vec2(t * 0.5, idx * 0.1));
    float y = cos(t * 0.8 + idx * 0.5) + snoise(vec2(t * 0.3, idx * 0.2));
    float z = sin(t * 0.3 + idx * 0.2) * cos(t * 0.5);
    
    // Warp based on alpha exhaustion
    float warp = 1.0 - uAlphaExhaustion;
    vec3 newPos = position;
    newPos.x += x * warp * 2.0;
    newPos.y += y * warp * 2.0;
    newPos.z += z * warp * 2.0;
    
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Color gradient: Cyan to Magenta based on depth and time
    float colorMix = (sin(t + idx) + 1.0) * 0.5;
    vec3 colorA = vec3(0.0, 1.0, 1.0); // Cyan
    vec3 colorB = vec3(1.0, 0.0, 1.0); // Magenta
    vColor = mix(colorA, colorB, colorMix);
    vAlpha = 0.8 - (distance(newPos, vec3(0.0)) * 0.1);
  }
`;

const attractorFragmentShader = `
  varying vec3 vColor;
  varying float vAlpha;
  
  void main() {
    if (vAlpha < 0.1) discard;
    gl_FragColor = vec4(vColor, vAlpha);
  }
`;

interface AttractorProps {
  alphaExhaustion: number;
  pointCount: number;
}

function AttractorMesh({ alphaExhaustion, pointCount }: AttractorProps) {
  const meshRef = useRef<THREE.Points>(null);
  const uniformRef = useRef<{ uTime: { value: number }; uAlphaExhaustion: { value: number } }>({
    uTime: { value: 0 },
    uAlphaExhaustion: { value: alphaExhaustion },
  });

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(pointCount * 3);
    const indices = new Float32Array(pointCount);
    
    for (let i = 0; i < pointCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 0.1;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 0.1;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 0.1;
      indices[i] = i;
    }
    
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aIndex', new THREE.BufferAttribute(indices, 1));
    return geo;
  }, [pointCount]);

  useFrame((state, delta) => {
    if (uniformRef.current) {
      uniformRef.current.uTime.value += delta;
      uniformRef.current.uAlphaExhaustion.value = alphaExhaustion;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1;
      meshRef.current.rotation.z += delta * 0.05;
    }
  });

  return (
    <points ref={meshRef} geometry={geometry}>
      <shaderMaterial
        vertexShader={attractorVertexShader}
        fragmentShader={attractorFragmentShader}
        uniforms={uniformRef.current}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export function OmegaPhaseSpace3D() {
  // Mock data: In production, this comes from WebSocket/Store
  const alphaExhaustion = 0.3; 
  const pointCount = 50000;

  return (
    <div className="relative w-full h-[600px] bg-[#0a0a0c] rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
        <h3 className="text-cyan-400 font-mono text-sm tracking-widest uppercase glow-text">
          Omega Phase Space
        </h3>
        <p className="text-cyan-700 font-mono text-xs">Takens Embedding / Poincaré Recurrence</p>
      </div>
      
      <Canvas camera={{ position: [0, 0, 8], fov: 60 }}>
        <color attach="background" args={['#0a0a0c']} />
        <fog attach="fog" args={['#0a0a0c', 5, 20]} />
        
        <AttractorMesh alphaExhaustion={alphaExhaustion} pointCount={pointCount} />
        <WickRotatedGhostRibbons alphaExhaustion={alphaExhaustion} />
        
        <EffectComposer>
          <Bloom luminanceThreshold={0.8} intensity={1.5} />
        </EffectComposer>
      </Canvas>
      
      <div className="absolute bottom-4 right-4 z-10">
        <LandauerThermodynamicGauge value={alphaExhaustion} />
      </div>
    </div>
  );
}
