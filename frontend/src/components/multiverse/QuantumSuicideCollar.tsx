'use client';

import React, { useRef, useEffect } from 'react';

interface QuantumSuicideCollarProps {
  survivalBranchId: number;
  isActive: boolean;
}

export function QuantumSuicideCollar({ survivalBranchId, isActive }: QuantumSuicideCollarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;

    const render = () => {
      time += 0.05;
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      if (!isActive) {
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#4a4a6a';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('QUANTUM COLLAR INACTIVE', width / 2, height / 2);
        return;
      }

      // Draw survival branch indicator
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.3;

      // Pulsing glow effect
      const pulse = 0.5 + 0.5 * Math.sin(time * 2);
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
      gradient.addColorStop(0, `rgba(255, 255, 255, ${0.8 * pulse})`);
      gradient.addColorStop(0.5, `rgba(0, 255, 255, ${0.4 * pulse})`);
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw collar ring
      ctx.strokeStyle = '#00ffff';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#00ffff';
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 0.7, 0, Math.PI * 2);
      ctx.stroke();

      // Draw branch ID
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'center';
      ctx.shadowColor = '#00ffff';
      ctx.shadowBlur = 10;
      ctx.fillText(`SURVIVAL BRANCH #${survivalBranchId}`, centerX, centerY - 10);
      ctx.font = '10px monospace';
      ctx.fillText('CONSCIOUSNESS LOCKED', centerX, centerY + 10);

      // Draw warning indicators
      const warningCount = 8;
      for (let i = 0; i < warningCount; i++) {
        const angle = (time + (i / warningCount) * Math.PI * 2) % (Math.PI * 2);
        const x = centerX + Math.cos(angle) * radius * 0.9;
        const y = centerY + Math.sin(angle) * radius * 0.9;

        ctx.fillStyle = i % 2 === 0 ? '#00ffff' : '#ff00ff';
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.shadowBlur = 0;
      animationFrameId = requestAnimationFrame(render);
    };

    const resizeObserver = new ResizeObserver(() => {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    });
    resizeObserver.observe(canvas);

    render();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      resizeObserver.disconnect();
    };
  }, [survivalBranchId, isActive]);

  return (
    <div className="relative w-full h-48 bg-[#0a0a0c] rounded-lg overflow-hidden border border-cyan-500/50">
      <div className="absolute top-2 left-2 z-10 pointer-events-none">
        <h4 className="text-cyan-400 font-mono text-xs tracking-wider uppercase">
          Quantum Suicide Collar
        </h4>
      </div>
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
}
