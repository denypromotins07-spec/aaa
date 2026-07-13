'use client';

import React, { useEffect, useRef, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { SwarmNode, SwarmLink, FORCE_SIMULATION_CONFIG, latencyToEdgeStyle, getNodeColor, calculateShatterEffect } from '../../utils/graph/forceSimulation';

interface SwarmTopologyGraphProps {
  width: number;
  height: number;
}

interface RawSwarmData {
  nodes: Array<{ id: string; role: string; latency: number; health: number }>;
  links: Array<{ source: string; target: string; latency: number; bandwidth: number }>;
}

export default function SwarmTopologyGraph({ width, height }: SwarmTopologyGraphProps) {
  const graphRef = useRef<any>(null);
  const [data, setData] = useState<{ nodes: SwarmNode[]; links: SwarmLink[] }>({ nodes: [], links: [] });
  const [fencedNodes, setFencedNodes] = useState<Set<string>>(new Set());
  const shatterTimersRef = useRef<Map<string, number>>(new Map());
  const engineTickThrottledRef = useRef<number>(0);

  // Initialize with mock swarm data
  useEffect(() => {
    const mockNodes: SwarmNode[] = [];
    const mockLinks: SwarmLink[] = [];
    const numNodes = 24;

    for (let i = 0; i < numNodes; i++) {
      mockNodes.push({
        id: `node-${i}`,
        role: i === 0 ? 'leader' : 'follower',
        latency: Math.random() * 100,
        health: 0.8 + Math.random() * 0.2,
        x: (Math.random() - 0.5) * 400,
        y: (Math.random() - 0.5) * 400,
        z: (Math.random() - 0.5) * 400,
      });
    }

    // Create mesh topology
    for (let i = 0; i < numNodes; i++) {
      for (let j = i + 1; j < numNodes; j++) {
        if (Math.random() > 0.7) {
          mockLinks.push({
            source: `node-${i}`,
            target: `node-${j}`,
            latency: Math.random() * 150,
            bandwidth: 1000 + Math.random() * 9000,
          });
        }
      }
    }

    setData({ nodes: mockNodes, links: mockLinks });

    // Simulate network partition and STONITH after 5 seconds
    const partitionTimer = setTimeout(() => {
      triggerStonithFencing('node-5');
    }, 5000);

    return () => clearTimeout(partitionTimer);
  }, []);

  const triggerStonithFencing = (nodeId: string) => {
    setFencedNodes((prev) => new Set(prev).add(nodeId));
    
    setData((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) =>
        n.id === nodeId ? { ...n, role: 'fenced', shatterProgress: 0 } : n
      ),
    }));

    // Animate shatter effect
    let progress = 0;
    const animateShatter = () => {
      progress += 0.02;
      if (progress >= 1) {
        // Remove node after shatter completes
        setData((prev) => ({
          nodes: prev.nodes.filter((n) => n.id !== nodeId),
          links: prev.links.filter(
            (l) => l.source !== nodeId && l.target !== nodeId
          ),
        }));
        shatterTimersRef.current.delete(nodeId);
        return;
      }

      setData((prev) => ({
        ...prev,
        nodes: prev.nodes.map((n) =>
          n.id === nodeId ? { ...n, shatterProgress: progress } : n
        ),
      }));

      shatterTimersRef.current.set(
        nodeId,
        requestAnimationFrame(animateShatter)
      );
    };

    shatterTimersRef.current.set(nodeId, requestAnimationFrame(animateShatter));
  };

  // Throttled engine tick handler to prevent main thread blocking
  const handleEngineTick = () => {
    const now = performance.now();
    // Throttle to 30fps (every ~33ms)
    if (now - engineTickThrottledRef.current < 33) return;
    engineTickThrottledRef.current = now;

    // Update graph visualization
    if (graphRef.current) {
      graphRef.current.graphData(data);
    }
  };

  // Cleanup shatter timers on unmount
  useEffect(() => {
    return () => {
      shatterTimersRef.current.forEach((timerId) => {
        cancelAnimationFrame(timerId);
      });
    };
  }, []);

  return (
    <div className="relative w-full h-full glass-panel rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-2 left-3 z-10 font-mono text-xs text-cyan-400 tracking-wider pointer-events-none">
        SWARM_TOPOLOGY :: RAFT_CONSENSUS
      </div>
      <div className="absolute bottom-2 left-3 z-10 font-mono text-xs text-yellow-400 pointer-events-none">
        NODES: {data.nodes.length} | LINKS: {data.links.length}
      </div>
      {fencedNodes.size > 0 && (
        <div className="absolute top-2 right-3 z-10 font-mono text-xs text-red-500 animate-pulse pointer-events-none">
          STONITH_ACTIVE :: {fencedNodes.size} NODES_FENCED
        </div>
      )}
      
      <div className="w-full h-full">
        <ForceGraph3D
          ref={graphRef}
          graphData={data}
          width={width}
          height={height}
          backgroundColor="rgba(10, 10, 12, 0)"
          nodeLabel={(node: any) => `${node.id}\nRole: ${node.role}\nLatency: ${node.latency.toFixed(1)}ms`}
          nodeColor={(node: any) => getNodeColor(node as SwarmNode)}
          nodeResolution={16}
          linkColor={(link: any) => latencyToEdgeStyle(link.latency).color}
          linkWidth={(link: any) => latencyToEdgeStyle(link.latency).width}
          linkOpacity={0.6}
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={0.005}
          linkDirectionalParticleColor={(link: any) => latencyToEdgeStyle(link.latency).color}
          onEngineTick={handleEngineTick}
          d3AlphaDecay={0.02}
          d3VelocityDecay={FORCE_SIMULATION_CONFIG.decay}
          warmupTicks={100}
          cooldownTicks={0}
          enableNodeDrag={false}
          showNavInfo={false}
        />
      </div>
    </div>
  );
}
