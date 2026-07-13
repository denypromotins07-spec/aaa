/**
 * WebGL Orderbook Heatmap Component
 * 
 * CRITICAL PERFORMANCE DESIGN:
 * - Renders at locked 60fps using requestAnimationFrame
 * - Completely decoupled from React's state update cycle
 * - Reads data directly from refs (no Zustand subscriptions in render loop)
 * - Uses offscreen canvas for double-buffering
 * - Texture-based rendering for maximum performance with large datasets
 */

'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { createShaderProgram, vertexShader, fragmentShader } from '@/utils/webgl/shaders';

interface OrderbookHeatmapProps {
  width?: number;
  height?: number;
  // Ref to latest telemetry data - read directly without React state
  latestFrameRef: React.MutableRefObject<{
    bids: [number, number][];
    asks: [number, number][];
    trades: [number, number, number][];
    health?: { latency_us: number };
  } | null>;
}

const HEATMAP_WIDTH = 512; // Internal rendering resolution
const HEATMAP_HEIGHT = 256;
const MAX_PRICE_LEVELS = 50;
const TIME_BUFFER_SIZE = 200; // Number of time slices to keep

export default function OrderbookHeatmap({ 
  width = 800, 
  height = 600,
  latestFrameRef 
}: OrderbookHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Heatmap data buffer - stores volume intensity over time
  // Format: Float32Array[TIME_BUFFER_SIZE * HEATMAP_HEIGHT]
  const heatmapDataRef = useRef<Float32Array>(
    new Float32Array(TIME_BUFFER_SIZE * HEATMAP_HEIGHT)
  );
  
  // Current write position in time buffer
  const timeIndexRef = useRef(0);
  
  // Previous mid-price for normalization
  const prevMidPriceRef = useRef<number>(0);
  
  // Initialize WebGL context and shaders
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const gl = canvas.getContext('webgl2', { 
      alpha: true,
      antialias: false,
      preserveDrawingBuffer: false,
    });
    
    if (!gl) {
      console.error('[HEATMAP] WebGL2 not supported');
      return;
    }
    
    glRef.current = gl;
    
    // Create shader program
    const program = createShaderProgram(gl, vertexShader, fragmentShader);
    if (!program) {
      console.error('[HEATMAP] Failed to create shader program');
      return;
    }
    programRef.current = program;
    
    // Create texture for heatmap data
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    // Initialize empty texture
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      TIME_BUFFER_SIZE,
      HEATMAP_HEIGHT,
      0,
      gl.RED,
      gl.FLOAT,
      heatmapDataRef.current
    );
    
    textureRef.current = texture;
    
    // Set up vertex buffer for full-screen quad
    const vertices = new Float32Array([
      // Positions     // UVs
      -1.0, -1.0,      0.0, 0.0,
       1.0, -1.0,      1.0, 0.0,
      -1.0,  1.0,      0.0, 1.0,
      -1.0,  1.0,      0.0, 1.0,
       1.0, -1.0,      1.0, 0.0,
       1.0,  1.0,      1.0, 1.0,
    ]);
    
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const uvLoc = gl.getAttribLocation(program, 'a_uv');
    
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0);
    
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);
    
    // Cleanup on unmount
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      gl.deleteTexture(texture);
      gl.deleteBuffer(vbo);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
    };
  }, []);
  
  // Update heatmap data from telemetry
  const updateHeatmapData = useCallback((frame: {
    bids: [number, number][];
    asks: [number, number][];
  }) => {
    const data = heatmapDataRef.current;
    const colOffset = timeIndexRef.current * HEATMAP_HEIGHT;
    
    // Calculate mid-price for normalization
    const bestBid = frame.bids.length > 0 ? frame.bids[0][0] : 0;
    const bestAsk = frame.asks.length > 0 ? frame.asks[0][0] : 0;
    const midPrice = (bestBid + bestAsk) / 2 || prevMidPriceRef.current;
    
    if (midPrice !== 0) {
      prevMidPriceRef.current = midPrice;
    }
    
    // Clear current column
    for (let y = 0; y < HEATMAP_HEIGHT; y++) {
      data[colOffset + y] = 0;
    }
    
    // Map price levels to Y positions
    const priceRange = midPrice * 0.02; // ±2% around mid-price
    const pricePerPixel = priceRange * 2 / HEATMAP_HEIGHT;
    
    // Render bids (below mid-price)
    frame.bids.slice(0, MAX_PRICE_LEVELS).forEach(([price, volume]) => {
      const priceOffset = midPrice - price;
      const y = Math.floor((priceRange - priceOffset) / pricePerPixel);
      
      if (y >= 0 && y < HEATMAP_HEIGHT / 2) {
        // Normalize volume (log scale for better visualization)
        const normalizedVolume = Math.log10(volume + 1) / 6; // Cap at ~1M
        data[colOffset + y] = Math.min(normalizedVolume, 1);
      }
    });
    
    // Render asks (above mid-price)
    frame.asks.slice(0, MAX_PRICE_LEVELS).forEach(([price, volume]) => {
      const priceOffset = price - midPrice;
      const y = Math.floor(HEATMAP_HEIGHT / 2 + priceOffset / pricePerPixel);
      
      if (y >= HEATMAP_HEIGHT / 2 && y < HEATMAP_HEIGHT) {
        const normalizedVolume = Math.log10(volume + 1) / 6;
        data[colOffset + y] = Math.min(normalizedVolume, 1);
      }
    });
    
    // Advance time index (circular buffer)
    timeIndexRef.current = (timeIndexRef.current + 1) % TIME_BUFFER_SIZE;
  }, []);
  
  // Render loop - runs at 60fps independent of React
  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    
    if (!gl || !program || !texture) return;
    
    let startTime = performance.now();
    
    const render = () => {
      const currentTime = (performance.now() - startTime) / 1000;
      
      // Check for new data
      const frame = latestFrameRef.current;
      if (frame && (frame.bids.length > 0 || frame.asks.length > 0)) {
        updateHeatmapData(frame);
        
        // Update texture with new column
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texSubImage2D(
          gl.TEXTURE_2D,
          0,
          0,
          0,
          TIME_BUFFER_SIZE,
          HEATMAP_HEIGHT,
          gl.RED,
          gl.FLOAT,
          heatmapDataRef.current
        );
      }
      
      // Clear and render
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
      gl.clearColor(0.04, 0.04, 0.05, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      
      gl.useProgram(program);
      
      // Set uniforms
      const timeLoc = gl.getUniformLocation(program, 'u_time');
      const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
      
      gl.uniform1f(timeLoc, currentTime);
      gl.uniform2f(resolutionLoc, gl.canvas.width, gl.canvas.height);
      
      // Draw full-screen quad
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [latestFrameRef, updateHeatmapData]);
  
  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full rounded"
        style={{ imageRendering: 'pixelated' }}
      />
      
      {/* Overlay labels */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 text-xs font-mono text-neon-cyan/50 writing-vertical-lr">
        ASKS ▲
      </div>
      <div className="absolute left-2 bottom-2 text-xs font-mono text-neon-magenta/50">
        BIDS ▼
      </div>
      <div className="absolute right-2 top-2 text-xs font-mono text-gray-500">
        TIME →
      </div>
    </div>
  );
}
