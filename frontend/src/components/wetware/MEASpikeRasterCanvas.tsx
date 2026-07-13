'use client';

import React, { useEffect, useRef } from 'react';
import { createSpikeBuffer, SpikePoint } from '../../utils/canvas/spikeBuffer';

interface MEASpikeRasterCanvasProps {
  electrodeCount: number;
  timeWindowMs: number;
  spikeData: SpikePoint[];
}

export function MEASpikeRasterCanvas({ 
  electrodeCount, 
  timeWindowMs,
  spikeData 
}: MEASpikeRasterCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>(0);
  const lastRenderTimeRef = useRef<number>(0);
  const bufferRef = useRef(createSpikeBuffer(50000));
  const startTimeRef = useRef<number>(Date.now());

  // Update buffer with new spike data
  useEffect(() => {
    if (spikeData.length > 0) {
      bufferRef.current.pushBatch(spikeData);
    }
  }, [spikeData]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // WebGL context lost handling
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      console.warn('WebGL context lost, will restore...');
    };

    const handleContextRestored = () => {
      console.log('WebGL context restored');
      startTimeRef.current = Date.now();
    };

    canvas.addEventListener('webglcontextlost', handleContextLost, false);
    canvas.addEventListener('webglcontextrestored', handleContextRestored, false);

    const render = (timestamp: number) => {
      // Throttle to ~60fps minimum interval
      if (timestamp - lastRenderTimeRef.current < 16) {
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }
      lastRenderTimeRef.current = timestamp;

      const width = rect.width;
      const height = rect.height;
      
      // Clear canvas with fade effect for trail
      ctx.fillStyle = 'rgba(10, 10, 12, 0.3)';
      ctx.fillRect(0, 0, width, height);

      const now = Date.now();
      const timeRange = timeWindowMs;
      const startTime = now - timeRange;

      // Get spikes in visible time window
      const spikes = bufferRef.current.getRange(startTime, now);
      
      // Calculate electrode spacing
      const electrodeHeight = height / Math.max(electrodeCount, 1);

      // Render each spike as a "firefly" particle
      for (const spike of spikes) {
        const x = ((spike.timestamp - startTime) / timeRange) * width;
        const y = (spike.electrodeId % electrodeCount) * electrodeHeight + electrodeHeight / 2;
        
        // Color based on spike type
        const isExcitatory = spike.type === 'excitatory';
        const hue = isExcitatory ? 140 : 300; // Green or Magenta
        const saturation = 80 + spike.amplitude * 20;
        const lightness = 50 + spike.amplitude * 30;
        
        // Draw spike as glowing particle
        ctx.beginPath();
        ctx.arc(x, y, 2 + spike.amplitude * 2, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, ${Math.min(1, spike.amplitude)})`;
        ctx.fill();
        
        // Add horizontal trail
        const trailLength = 20 + spike.amplitude * 30;
        const gradient = ctx.createLinearGradient(x - trailLength, y, x, y);
        gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);
        gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.3)`);
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x - trailLength, y - 1, trailLength, 2);
      }

      // Draw electrode labels (every 10th)
      ctx.fillStyle = '#4a5568';
      ctx.font = '10px JetBrains Mono, monospace';
      for (let i = 0; i < electrodeCount; i += 10) {
        const y = i * electrodeHeight + electrodeHeight / 2;
        ctx.fillText(`E${i}`, 5, y + 3);
      }

      // Draw time axis
      ctx.strokeStyle = '#2d3748';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height - 20);
      ctx.lineTo(width, height - 20);
      ctx.stroke();

      // Time markers
      ctx.fillStyle = '#4a5568';
      for (let t = 0; t <= timeRange; t += timeRange / 5) {
        const x = (t / timeRange) * width;
        const timeLabel = `${(t / 1000).toFixed(1)}s`;
        ctx.fillText(timeLabel, x - 15, height - 5);
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      canvas.removeEventListener('webglcontextlost', handleContextLost);
      canvas.removeEventListener('webglcontextrestored', handleContextRestored);
    };
  }, [electrodeCount, timeWindowMs]);

  return (
    <div className="relative w-full h-[300px] bg-[#0a0a0c] rounded-lg overflow-hidden border border-cyan-900/30">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: 'block' }}
      />
      <div className="absolute top-2 right-2 px-2 py-1 bg-black/50 backdrop-blur-sm rounded text-xs font-mono text-cyan-400">
        Spikes: {bufferRef.current.getCount()}
      </div>
    </div>
  );
}
