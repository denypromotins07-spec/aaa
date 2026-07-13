'use client';

import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { RADAR_VERTEX_SHADER, RADAR_FRAGMENT_SHADER } from '../../utils/webgl/radarShaders';

interface ConstitutionMetric {
  name: string;
  value: number; // 0.0 to 1.0
  threshold: number;
}

interface ConstitutionalRadarProps {
  metrics: ConstitutionMetric[];
  isAligned: boolean;
}

function RadarMesh({ metrics, isAligned }: { metrics: ConstitutionMetric[]; isAligned: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const geometryRef = useRef<THREE.ShapeGeometry | null>(null);
  const timeRef = useRef(0);

  const polygonPoints = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const segments = Math.min(6, metrics.length);
    
    for (let i = 0; i < segments; i++) {
      const metric = metrics[i] || { value: 0.5 };
      const angle = (i / segments) * Math.PI * 2 - Math.PI / 2;
      const radius = metric.value * 0.4;
      
      points.push(new THREE.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        0
      ));
    }
    
    return points;
  }, [metrics]);

  // Create geometry once and update it
  useEffect(() => {
    if (polygonPoints.length > 0) {
      const shape = new THREE.Shape();
      shape.moveTo(polygonPoints[0].x, polygonPoints[0].y);
      for (let i = 1; i < polygonPoints.length; i++) {
        shape.lineTo(polygonPoints[i].x, polygonPoints[i].y);
      }
      shape.closePath();
      
      if (geometryRef.current) {
        geometryRef.current.dispose();
      }
      geometryRef.current = new THREE.ShapeGeometry(shape);
      
      if (meshRef.current) {
        meshRef.current.geometry = geometryRef.current;
      }
    }
    
    return () => {
      if (geometryRef.current) {
        geometryRef.current.dispose();
      }
    };
  }, [polygonPoints]);

  useFrame((state, delta) => {
    timeRef.current += delta;
    
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = timeRef.current;
      materialRef.current.uniforms.uPulse.value = isAligned ? 0.3 : 1.0;
      materialRef.current.uniforms.uColor.value = isAligned 
        ? new THREE.Color('#00ffff') 
        : new THREE.Color('#ff0055');
    }
  });

  // Cleanup material on unmount
  useEffect(() => {
    return () => {
      if (materialRef.current) {
        materialRef.current.dispose();
      }
    };
  }, []);

  return (
    <mesh ref={meshRef} rotation={[Math.PI / 2, 0, 0]}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={RADAR_VERTEX_SHADER}
        fragmentShader={RADAR_FRAGMENT_SHADER}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
        uniforms={{
          uTime: { value: 0 },
          uPulse: { value: 0.3 },
          uColor: { value: new THREE.Color('#00ffff') },
          uGlow: { value: 0.8 },
          uAxisCount: { value: 6 },
        }}
      />
    </mesh>
  );
}

function AxisLines() {
  const linesRef = useRef<THREE.LineSegments>(null);
  
  const geometry = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const segments = 6;
    
    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2 - Math.PI / 2;
      const x = Math.cos(angle) * 0.5;
      const y = Math.sin(angle) * 0.5;
      
      points.push(new THREE.Vector3(0, 0, 0));
      points.push(new THREE.Vector3(x, y, 0));
    }
    
    return new THREE.BufferGeometry().setFromPoints(points);
  }, []);

  return (
    <lineSegments ref={linesRef} geometry={geometry}>
      <lineBasicMaterial color="#00ffff" opacity={0.4} transparent />
    </lineSegments>
  );
}

function ConcentricRings() {
  const ringsRef = useRef<THREE.Group>(null);
  
  const rings = useMemo(() => {
    return [0.1, 0.2, 0.3, 0.4, 0.5].map((radius, i) => (
      <mesh key={i} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 0.98, radius, 64]} />
        <meshBasicMaterial color="#00ffff" opacity={0.15} transparent side={THREE.DoubleSide} />
      </mesh>
    ));
  }, []);

  return <group ref={ringsRef}>{rings}</group>;
}

export function ConstitutionalRadar3D({ metrics, isAligned }: ConstitutionalRadarProps) {
  return (
    <div className="w-full h-[400px] relative">
      <Canvas camera={{ position: [0, 0, 1.5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <ConcentricRings />
        <AxisLines />
        <RadarMesh metrics={metrics} isAligned={isAligned} />
      </Canvas>
      
      {/* Holographic overlay effect */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-cyan-500/5 to-transparent mix-blend-screen" />
    </div>
  );
}
