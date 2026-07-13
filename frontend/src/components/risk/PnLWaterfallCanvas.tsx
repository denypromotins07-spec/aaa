'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { renderPnLWaterfall, RenderConfig } from '../../utils/canvas/chartRenderers';

interface PnLWaterfallCanvasProps {
  data: number[];
  width?: number;
  height?: number;
}

export default function PnLWaterfallCanvas({ 
  data, 
  width = 800, 
  height = 300 
}: PnLWaterfallCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number>(0);
  const dataRef = useRef<number[]>(data);
  const lastRenderTimeRef = useRef<number>(0);

  // Update data ref without triggering re-render
  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  // Initialize offscreen canvas for double-buffering
  useEffect(() => {
    offscreenCanvasRef.current = document.createElement('canvas');
    offscreenCanvasRef.current.width = width;
    offscreenCanvasRef.current.height = height;

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height]);

  // Render loop using requestAnimationFrame
  const render = useCallback((timestamp: number) => {
    // Throttle to ~60fps max
    if (timestamp - lastRenderTimeRef.current < 16) {
      animationFrameRef.current = requestAnimationFrame(render);
      return;
    }
    lastRenderTimeRef.current = timestamp;

    const canvas = canvasRef.current;
    const offscreen = offscreenCanvasRef.current;
    
    if (!canvas || !offscreen) return;

    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const config: RenderConfig = {
      width,
      height,
      padding: { top: 20, right: 20, bottom: 30, left: 50 },
    };

    // Determine if currently in profit (latest value > 0)
    const currentData = dataRef.current;
    const latestValue = currentData.length > 0 ? currentData[currentData.length - 1] : 0;
    const isProfit = latestValue >= 0;

    // Render to offscreen canvas
    renderPnLWaterfall(ctx, currentData, config, isProfit);

    // Blit to main canvas (double-buffering)
    const mainCtx = canvas.getContext('2d');
    if (mainCtx) {
      mainCtx.clearRect(0, 0, width, height);
      mainCtx.drawImage(offscreen, 0, 0);
    }

    animationFrameRef.current = requestAnimationFrame(render);
  }, [width, height]);

  // Start render loop
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
        style={{ display: 'block' }}
      />
      
      {/* Overlay stats */}
      <div className="absolute top-4 right-4 font-mono text-xs space-y-1 pointer-events-none">
        <div className="text-gray-400">
          CUMULATIVE PnL
        </div>
        <div className={`text-lg ${
          data[data.length - 1] >= 0 ? 'text-cyan-400' : 'text-magenta-400'
        } drop-shadow-[0_0_8px_currentColor]`}>
          ${(data[data.length - 1] || 0).toFixed(2)}
        </div>
      </div>
    </div>
  );
}
