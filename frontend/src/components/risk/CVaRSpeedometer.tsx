'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { renderCVaRSpeedometer, RenderConfig } from '../../utils/canvas/chartRenderers';

interface CVaRSpeedometerProps {
  currentValue: number;
  maxValue?: number;
  criticalThreshold?: number;
  width?: number;
  height?: number;
}

export default function CVaRSpeedometer({
  currentValue,
  maxValue = 0.1,
  criticalThreshold = 0.05,
  width = 300,
  height = 250,
}: CVaRSpeedometerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number>(0);
  const valueRef = useRef<number>(currentValue);
  const lastRenderTimeRef = useRef<number>(0);

  // Update value ref without triggering re-render
  useEffect(() => {
    valueRef.current = currentValue;
  }, [currentValue]);

  // Initialize offscreen canvas
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

  // Render loop
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
      padding: { top: 20, right: 20, bottom: 40, left: 20 },
    };

    // Render to offscreen canvas
    renderCVaRSpeedometer(
      ctx,
      valueRef.current,
      maxValue,
      criticalThreshold,
      config
    );

    // Blit to main canvas
    const mainCtx = canvas.getContext('2d');
    if (mainCtx) {
      mainCtx.clearRect(0, 0, width, height);
      mainCtx.drawImage(offscreen, 0, 0);
    }

    animationFrameRef.current = requestAnimationFrame(render);
  }, [width, height, maxValue, criticalThreshold]);

  // Start render loop
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  const isCritical = currentValue >= criticalThreshold;

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full max-w-[300px] max-h-[250px]"
        style={{ display: 'block' }}
      />

      {/* Critical warning overlay */}
      {isCritical && (
        <div className="absolute inset-0 pointer-events-none animate-pulse">
          <div className="absolute inset-0 border-2 border-red-500/30 rounded-lg" />
        </div>
      )}

      {/* Info panel below gauge */}
      <div className="mt-2 text-center font-mono text-xs space-y-1">
        <div className={`text-sm ${isCritical ? 'text-red-400 animate-pulse' : 'text-gray-400'}`}>
          {isCritical ? '⚠ CRITICAL RISK LEVEL ⚠' : 'RISK SURFACE GAUGE'}
        </div>
        <div className="flex gap-4 justify-center text-gray-500">
          <span>MAX: {maxValue.toFixed(4)}</span>
          <span>LIMIT: {criticalThreshold.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}
