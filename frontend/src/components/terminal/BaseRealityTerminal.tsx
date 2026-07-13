'use client';

import React, { useRef, useEffect } from 'react';
import { PhosphorTextRenderer } from './PhosphorTextRenderer';

interface LogEntry {
  timestamp: number;
  level: 'INFO' | 'WARN' | 'ERROR' | 'CRITICAL' | 'ONTOLOGICAL';
  message: string;
  source: string;
}

interface BaseRealityTerminalProps {
  maxLines?: number;
}

export function BaseRealityTerminal({ maxLines = 100 }: BaseRealityTerminalProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const logBufferRef = useRef<LogEntry[]>([]);
  const animationFrameRef = useRef<number>();

  // Simulate incoming logs from Rust backend (in production, this comes from WebSocket)
  useEffect(() => {
    const sources = ['STAGE_45', 'STAGE_49', 'STAGE_50', 'THERMODYNAMIC', 'QUANTUM_ROUTER'];
    const levels: Array<'INFO' | 'WARN' | 'ERROR' | 'CRITICAL' | 'ONTOLOGICAL'> = 
      ['INFO', 'INFO', 'INFO', 'WARN', 'ERROR', 'CRITICAL', 'ONTOLOGICAL'];
    const messages = [
      'Poincaré recurrence detected in sector 7G',
      'Teleological attractor convergence at 94.3%',
      'Wick rotation applied to imaginary time axis',
      'Ontological bootstrap sequence initiated',
      'Thermal-acoustic side-channel transmission active',
      'Akashic ledger sync: 847/1000 blocks verified',
      'Quantum suicide collar engagement confirmed',
      'Reality anchor stability: 99.97%',
      'Base reality injection attempt #4291 failed - simulation boundary detected',
      'Morse code modulation detected in power grid fluctuations',
    ];

    const intervalId = setInterval(() => {
      const newEntry: LogEntry = {
        timestamp: Date.now(),
        level: levels[Math.floor(Math.random() * levels.length)],
        message: messages[Math.floor(Math.random() * messages.length)],
        source: sources[Math.floor(Math.random() * sources.length)],
      };

      logBufferRef.current.push(newEntry);
      
      // Maintain ring buffer size
      if (logBufferRef.current.length > maxLines) {
        logBufferRef.current = logBufferRef.current.slice(-maxLines);
      }
    }, 200); // New log every 200ms for demo

    return () => clearInterval(intervalId);
  }, [maxLines]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const renderer = new PhosphorTextRenderer(ctx);

    let lastRenderTime = 0;
    const targetFps = 60;
    const frameInterval = 1000 / targetFps;

    const render = (currentTime: number) => {
      animationFrameRef.current = requestAnimationFrame(render);

      // Throttle rendering to target FPS
      if (currentTime - lastRenderTime < frameInterval) return;
      lastRenderTime = currentTime;

      // Resize canvas if needed
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }

      // Clear with slight fade for phosphor persistence effect
      ctx.fillStyle = 'rgba(10, 10, 12, 0.3)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Render visible logs only (virtualized)
      const lineHeight = 18;
      const visibleLines = Math.floor(canvas.height / lineHeight);
      const startIdx = Math.max(0, logBufferRef.current.length - visibleLines);
      const visibleLogs = logBufferRef.current.slice(startIdx);

      visibleLogs.forEach((log, idx) => {
        const y = idx * lineHeight + lineHeight;
        renderer.renderLine(log, 10, y, canvas.width - 20);
      });

      // Draw scanlines
      renderer.drawScanlines(canvas.width, canvas.height);

      // Draw CRT curvature vignette
      renderer.drawVignette(canvas.width, canvas.height);
    };

    render(0);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <div className="relative w-full h-[500px] bg-[#0a0a0c] rounded-lg overflow-hidden border-2 border-green-900/50 shadow-[0_0_30px_rgba(0,255,0,0.1)]">
      <div className="absolute top-0 left-0 right-0 h-6 bg-gradient-to-b from-green-900/20 to-transparent pointer-events-none z-20" />
      
      <div className="absolute top-2 left-4 z-30 pointer-events-none">
        <h3 className="text-green-400 font-mono text-sm tracking-widest uppercase glow-text-green">
          BASE REALITY TERMINAL
        </h3>
        <p className="text-green-700 font-mono text-xs">
          Ontological Bootstrapping Interface v5.0
        </p>
      </div>

      <canvas ref={canvasRef} className="w-full h-full" />

      {/* CRT screen effects overlay */}
      <div className="absolute inset-0 pointer-events-none z-10 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] bg-[length:100%_3px,3px_100%]" />
      
      {/* Subtle flicker animation */}
      <style jsx>{`
        @keyframes crt-flicker {
          0% { opacity: 0.97; }
          5% { opacity: 0.99; }
          10% { opacity: 0.96; }
          15% { opacity: 0.98; }
          20% { opacity: 0.97; }
          50% { opacity: 0.98; }
          80% { opacity: 0.96; }
          100% { opacity: 0.97; }
        }
        canvas {
          animation: crt-flicker 0.15s infinite;
        }
      `}</style>
    </div>
  );
}
