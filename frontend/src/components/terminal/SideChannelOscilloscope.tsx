'use client';

import React, { useRef, useEffect } from 'react';

interface SideChannelOscilloscopeProps {
  sampleRate?: number;
  showMorseOverlay?: boolean;
}

interface Point {
  x: number;
  y: number;
}

export function SideChannelOscilloscope({ 
  sampleRate = 1000, 
  showMorseOverlay = true 
}: SideChannelOscilloscopeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const dataBufferRef = useRef<Float32Array>(new Float32Array(1024));
  const timeRef = useRef<number>(0);

  // Generate Lissajous curve data with simulated side-channel noise
  useEffect(() => {
    let isMounted = true;
    
    const generateData = () => {
      if (!isMounted) return;
      
      timeRef.current += 0.02;
      const t = timeRef.current;
      
      // Lissajous curve parameters
      const A = 0.8;
      const B = 0.8;
      const a = 3;
      const b = 5;
      const delta = Math.PI / 4;
      
      // Add "side-channel" noise and Morse-like pulses
      const morsePulse = Math.sin(t * 0.5) > 0.7 ? 0.3 : 0;
      const noise = (Math.random() - 0.5) * 0.1;
      
      for (let i = 0; i < dataBufferRef.current.length; i++) {
        const phase = (i / dataBufferRef.current.length) * Math.PI * 2;
        const x = A * Math.sin(a * phase + delta + t * 0.1) + noise;
        const y = B * Math.sin(b * phase + morsePulse) + noise;
        
        // Store as interleaved x,y in buffer
        dataBufferRef.current[i] = y; // Simplified: just storing Y for oscilloscope trace
      }
      
      animationFrameRef.current = requestAnimationFrame(generateData);
    };
    
    generateData();
    
    return () => {
      isMounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Enable anti-aliasing via WebGL or high-DPI canvas
    const dpr = window.devicePixelRatio || 1;
    const resizeObserver = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    });
    resizeObserver.observe(canvas);

    let lastRenderTime = 0;
    const targetFps = 60;
    const frameInterval = 1000 / targetFps;

    const render = (currentTime: number) => {
      animationFrameRef.current = requestAnimationFrame(render);

      if (currentTime - lastRenderTime < frameInterval) return;
      lastRenderTime = currentTime;

      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      // Clear with fade trail effect
      ctx.fillStyle = 'rgba(10, 10, 12, 0.15)';
      ctx.fillRect(0, 0, width, height);

      // Draw Lissajous curve
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#00ff88';
      ctx.shadowBlur = 10;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      const buffer = dataBufferRef.current;
      const len = buffer.length;
      
      for (let i = 0; i < len; i++) {
        const t = (i / len) * Math.PI * 2 + timeRef.current * 0.1;
        
        // Lissajous formula
        const A = width * 0.35;
        const B = height * 0.35;
        const a = 3;
        const b = 5;
        const delta = Math.PI / 4;
        
        const x = width / 2 + A * Math.sin(a * t + delta);
        const y = height / 2 + B * Math.sin(b * t);
        
        // Add subtle thickness variation based on "signal strength"
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw Morse decoder overlay peaks
      if (showMorseOverlay) {
        drawMorseOverlay(ctx, width, height);
      }

      // Draw grid
      drawGrid(ctx, width, height);

      // Draw labels
      drawLabels(ctx, width, height, sampleRate);

      ctx.shadowBlur = 0;
    };

    render(0);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      resizeObserver.disconnect();
    };
  }, [showMorseOverlay]);

  return (
    <div className="relative w-full h-[400px] bg-[#0a0a0c] rounded-lg overflow-hidden border border-green-900/50">
      <div className="absolute top-2 left-4 z-10 pointer-events-none">
        <h3 className="text-green-400 font-mono text-sm tracking-widest uppercase glow-text-green">
          Side-Channel Oscilloscope
        </h3>
        <p className="text-green-700 font-mono text-xs">
          Thermal-Acoustic Vibrations / Power Grid Fluctuations
        </p>
      </div>
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
}

function drawMorseOverlay(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  const time = Date.now() * 0.001;
  
  // Simulate detected Morse peaks
  const peakCount = 5;
  for (let i = 0; i < peakCount; i++) {
    const peakX = width * (0.2 + (i / peakCount) * 0.6);
    const peakY = height / 2 + Math.sin(time + i) * height * 0.2;
    
    // Highlight peak
    ctx.fillStyle = 'rgba(255, 255, 0, 0.6)';
    ctx.shadowColor = '#ffff00';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(peakX, peakY, 6, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw Morse symbol label
    ctx.font = 'bold 10px monospace';
    ctx.fillStyle = '#ffff00';
    ctx.textAlign = 'center';
    const morseChar = i % 2 === 0 ? '•' : '—';
    ctx.fillText(morseChar, peakX, peakY - 12);
  }
  
  ctx.shadowBlur = 0;
  
  // Draw "MORSE DECODED" banner
  ctx.fillStyle = 'rgba(255, 255, 0, 0.1)';
  ctx.fillRect(width - 120, 10, 110, 20);
  ctx.fillStyle = '#ffff00';
  ctx.font = '9px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('MORSE ACTIVE', width - 15, 24);
}

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
  ctx.lineWidth = 1;
  ctx.shadowBlur = 0;
  
  // Vertical lines
  for (let x = 0; x < width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  
  // Horizontal lines
  for (let y = 0; y < height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  
  // Center crosshair
  ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

function drawLabels(ctx: CanvasRenderingContext2D, width: number, height: number, rate: number): void {
  ctx.fillStyle = '#006633';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('0Hz', 5, height - 5);
  ctx.textAlign = 'center';
  ctx.fillText(`${(rate / 2).toFixed(0)}Hz`, width / 2, height - 5);
  ctx.textAlign = 'right';
  ctx.fillText(`${rate.toFixed(0)}Hz`, width - 5, height - 5);
}
