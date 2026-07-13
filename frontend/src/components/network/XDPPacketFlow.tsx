'use client';

import React, { useEffect, useRef } from 'react';
import { ParticlePool, renderParticles, spawnXDPPacket, PARTICLE_STYLES } from '../../utils/canvas/particleSystems';

interface XDPPacketFlowProps {
  width: number;
  height: number;
}

export default function XDPPacketFlow({ width, height }: XDPPacketFlowProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlePoolRef = useRef<ParticlePool | null>(null);
  const animationFrameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  // Network flow configuration
  const flowsRef = useRef<Array<{
    startX: number;
    startY: number;
    endX: number;
    endY: number;
    rate: number; // packets per second
    isSplice: boolean;
    lastSpawn: number;
  }>>([]);

  useEffect(() => {
    // Initialize particle pool
    particlePoolRef.current = new ParticlePool(500);

    // Define network flow paths
    flowsRef.current = [
      // Exchange -> SmartNIC (incoming UDP)
      { startX: 50, startY: height * 0.2, endX: width * 0.4, endY: height * 0.5, rate: 30, isSplice: false, lastSpawn: 0 },
      { startX: 50, startY: height * 0.5, endX: width * 0.4, endY: height * 0.5, rate: 25, isSplice: false, lastSpawn: 0 },
      { startX: 50, startY: height * 0.8, endX: width * 0.4, endY: height * 0.5, rate: 20, isSplice: false, lastSpawn: 0 },
      
      // SmartNIC -> XDP Filter (splice point)
      { startX: width * 0.4, startY: height * 0.5, endX: width * 0.6, endY: height * 0.5, rate: 15, isSplice: true, lastSpawn: 0 },
      
      // XDP -> Dark Pool (outgoing)
      { startX: width * 0.6, startY: height * 0.5, endX: width - 50, endY: height * 0.3, rate: 10, isSplice: false, lastSpawn: 0 },
      { startX: width * 0.6, startY: height * 0.5, endX: width - 50, endY: height * 0.7, rate: 8, isSplice: false, lastSpawn: 0 },
    ];

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      particlePoolRef.current?.reset();
    };
  }, [width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const pool = particlePoolRef.current;
    if (!pool) return;

    const render = (timestamp: number) => {
      const deltaTime = (timestamp - lastTimeRef.current) / 1000;
      lastTimeRef.current = timestamp;

      // Cap delta time to prevent explosion on tab switch
      const dt = Math.min(deltaTime, 0.1);

      // Clear canvas with slight fade for motion blur
      ctx.fillStyle = 'rgba(10, 10, 12, 0.2)';
      ctx.fillRect(0, 0, width, height);

      // Spawn new particles based on flow rates
      const now = timestamp;
      flowsRef.current.forEach((flow) => {
        const interval = 1000 / flow.rate;
        if (now - flow.lastSpawn >= interval) {
          spawnXDPPacket(
            pool,
            flow.startX,
            flow.startY,
            flow.endX,
            flow.endY,
            flow.isSplice
          );
          flow.lastSpawn = now;
        }
      });

      // Update and render particles
      pool.update(dt);
      const activeParticles = pool.getActiveParticles();
      renderParticles(ctx, activeParticles);

      // Draw network path indicators
      drawNetworkPaths(ctx, width, height);

      // Draw stats
      drawStats(ctx, pool.getActiveCount());

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render(0);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height]);

  return (
    <div className="relative w-full h-full glass-panel rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-2 left-3 z-10 font-mono text-xs text-cyan-400 tracking-wider pointer-events-none">
        XDP_PACKET_FLOW :: SMARTNIC_EBPF
      </div>
      <div className="absolute bottom-2 right-3 z-10 font-mono text-xs text-purple-400 pointer-events-none">
        UDP_INGRESS :: TCP_PIGGYBACK :: DARK_POOL_EGRESS
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
    </div>
  );
}

function drawNetworkPaths(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);

  // Draw exchange zone
  ctx.strokeRect(10, height * 0.1, 80, height * 0.8);
  ctx.fillStyle = 'rgba(6, 182, 212, 0.1)';
  ctx.fillRect(10, height * 0.1, 80, height * 0.8);
  
  ctx.fillStyle = '#06b6d4';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('EXCHANGE', 20, height * 0.15);
  ctx.fillText('UDP IN', 20, height * 0.85);

  // Draw SmartNIC/XDP zone
  const smartNicX = width * 0.4 - 30;
  ctx.strokeRect(smartNicX, height * 0.4, 120, height * 0.2);
  ctx.fillStyle = 'rgba(232, 121, 249, 0.1)';
  ctx.fillRect(smartNicX, height * 0.4, 120, height * 0.2);
  
  ctx.fillStyle = '#e879f9';
  ctx.fillText('SMARTNIC', smartNicX + 10, height * 0.45);
  ctx.fillText('XDP FILTER', smartNicX + 10, height * 0.58);

  // Draw Dark Pool zone
  ctx.strokeRect(width - 90, height * 0.2, 80, height * 0.6);
  ctx.fillStyle = 'rgba(34, 197, 94, 0.1)';
  ctx.fillRect(width - 90, height * 0.2, 80, height * 0.6);
  
  ctx.fillStyle = '#22c55e';
  ctx.fillText('DARK POOL', width - 80, height * 0.25);
  ctx.fillText('RFQ', width - 80, height * 0.75);
}

function drawStats(ctx: CanvasRenderingContext2D, activeCount: number) {
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(`ACTIVE_PARTICLES: ${activeCount}`, 10, 20);
  ctx.textAlign = 'left';
}
