'use client';

import React, { useRef, useEffect } from 'react';

interface FreeEnergyHeartbeatProps {
  freeEnergy: number; // Variational Free Energy / Surprise value
  threshold: number;
  maxDisplayValue?: number;
}

export function FreeEnergyHeartbeat({ 
  freeEnergy, 
  threshold,
  maxDisplayValue = 10 
}: FreeEnergyHeartbeatProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>(0);
  const phaseRef = useRef<number>(0);
  const lastValueRef = useRef<number>(freeEnergy);

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

    const render = () => {
      const width = rect.width;
      const height = rect.height;
      const centerX = width / 2;
      const centerY = height / 2;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Smooth transition for value
      lastValueRef.current = lastValueRef.current * 0.9 + freeEnergy * 0.1;
      const normalizedValue = Math.min(lastValueRef.current / maxDisplayValue, 1);

      // Determine color based on stress level
      let baseColor = { r: 0, g: 255, b: 136 }; // Green (optimal)
      if (normalizedValue > 0.5) {
        // Interpolate to amber
        const t = (normalizedValue - 0.5) * 2;
        baseColor = {
          r: Math.floor(255 * t),
          g: Math.floor(200 * (1 - t) + 255 * t),
          b: Math.floor(136 * (1 - t)),
        };
      }
      if (normalizedValue > 0.8) {
        // Interpolate to red
        const t = (normalizedValue - 0.8) * 5;
        baseColor = {
          r: 255,
          g: Math.floor(255 * (1 - t)),
          b: Math.floor(136 * (1 - t)),
        };
      }

      const colorString = `rgb(${baseColor.r}, ${baseColor.g}, ${baseColor.b})`;

      // Update phase for heartbeat animation
      phaseRef.current += 0.05 + normalizedValue * 0.1;
      const pulse = Math.sin(phaseRef.current) * 0.1 + 0.15;

      // Draw outer glow ring
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, height / 2);
      gradient.addColorStop(0, `${colorString}40`);
      gradient.addColorStop(0.5, `${colorString}20`);
      gradient.addColorStop(1, 'transparent');
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, height / 2, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Draw pulsating "heartbeat" circle
      const baseRadius = Math.min(width, height) * 0.25;
      const currentRadius = baseRadius * (1 + pulse * normalizedValue);
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, currentRadius, 0, Math.PI * 2);
      ctx.fillStyle = `${colorString}60`;
      ctx.fill();

      // Draw inner core
      ctx.beginPath();
      ctx.arc(centerX, centerY, currentRadius * 0.7, 0, Math.PI * 2);
      ctx.fillStyle = colorString;
      ctx.fill();

      // Draw threshold indicator ring
      const thresholdRadius = baseRadius * (threshold / maxDisplayValue);
      ctx.beginPath();
      ctx.arc(centerX, centerY, thresholdRadius, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff40';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw value text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 24px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(lastValueRef.current.toFixed(2), centerX, centerY - 10);

      ctx.fillStyle = '#9ca3af';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.fillText('FREE ENERGY', centerX, centerY + 20);

      // Status indicator
      const status = normalizedValue > 0.8 ? 'CRITICAL' : normalizedValue > 0.5 ? 'ELEVATED' : 'OPTIMAL';
      ctx.fillStyle = normalizedValue > 0.8 ? '#ef4444' : normalizedValue > 0.5 ? '#fbbf24' : '#00ff88';
      ctx.font = 'bold 14px JetBrains Mono, monospace';
      ctx.fillText(status, centerX, centerY + 45);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [freeEnergy, threshold, maxDisplayValue]);

  return (
    <div className="relative w-full h-[250px] bg-[#0a0a0c]/80 backdrop-blur-md rounded-lg border border-cyan-900/30 p-4">
      <div className="absolute top-3 left-4 text-xs text-gray-500 font-mono uppercase tracking-wider">
        Variational Free Energy
      </div>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: 'block' }}
      />
    </div>
  );
}
