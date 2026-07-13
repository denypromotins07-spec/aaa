'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { useMarketDataRef } from '@/store/nexusStore';
import { 
  createProgram, 
  HEATMAP_FRAGMENT_SHADER, 
  VERTEX_SHADER 
} from '@/utils/webgl/shaders';

interface WebGLHeatmapProps {
  width?: number;
  height?: number;
  className?: string;
}

/**
 * High-performance WebGL Orderbook Heatmap
 * 
 * CRITICAL PERFORMANCE FEATURES:
 * - Renders at locked 60fps using requestAnimationFrame
 * - Completely decoupled from React state updates
 * - Uses texture-based rendering for efficient memory usage
 * - Y-axis: Price (bids bottom, asks top)
 * - X-axis: Time (scrolling)
 * - Color intensity: Volume/liquidity
 */
export function WebGLHeatmap({ 
  width = 800, 
  height = 600,
  className = ''
}: WebGLHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const marketDataRef = useMarketDataRef();
  
  // Rendering state (kept outside React)
  const renderState = useRef({
    time: 0,
    scrollOffset: 0,
    heatmapData: new Float32Array(256 * 256), // 256x256 heatmap texture
    midPrice: 0,
    priceRange: 100,
    colorScheme: 0, // 0=cyberpunk, 1=matrix, 2=mono
  });

  /**
   * Initialize WebGL context and shaders
   */
  const initWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl2', { 
      alpha: true,
      antialias: false,
      powerPreference: 'high-performance',
    });
    
    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    glRef.current = gl;

    // Create shader program
    const program = createProgram(gl, VERTEX_SHADER, HEATMAP_FRAGMENT_SHADER);
    if (!program) {
      console.error('Failed to create shader program');
      return;
    }
    programRef.current = program;

    // Create full-screen quad vertices
    const vertices = new Float32Array([
      -1, -1,  0, 0,
       1, -1,  1, 0,
       1,  1,  1, 1,
      -1, -1,  0, 0,
       1,  1,  1, 1,
      -1,  1,  0, 1,
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

    // Create heatmap texture
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    // Initialize empty texture
    gl.texImage2D(
      gl.TEXTURE_2D, 
      0, 
      gl.R32F, 
      256, 
      256, 
      0, 
      gl.RED, 
      gl.FLOAT, 
      renderState.current.heatmapData
    );
    
    textureRef.current = texture;

    // Set clear color (obsidian background)
    gl.clearColor(0.04, 0.04, 0.05, 1.0);
  }, []);

  /**
   * Update heatmap texture with new orderbook data
   * Called via callback from WebSocket hook
   */
  const updateHeatmapData = useCallback((data: unknown) => {
    const marketData = data as {
      l2Bids: [number, number][];
      l2Asks: [number, number][];
      bestBid: number;
      bestAsk: number;
    };

    if (!marketData?.l2Bids || !marketData.l2Asks) return;

    const state = renderState.current;
    
    // Update mid-price
    state.midPrice = (marketData.bestBid + marketData.bestAsk) / 2;
    state.priceRange = Math.max(
      marketData.bestAsk - marketData.bestBid,
      50 // Minimum range
    );

    // Shift existing data left (time scroll)
    const heatmapSize = 256;
    const newData = new Float32Array(heatmapSize * heatmapSize);
    
    for (let y = 0; y < heatmapSize; y++) {
      for (let x = 0; x < heatmapSize - 1; x++) {
        newData[y * heatmapSize + x] = state.heatmapData[y * heatmapSize + x + 1];
      }
    }

    // Add new column on the right
    const allLevels = [...marketData.l2Bids.reverse(), ...marketData.l2Asks];
    const maxVolume = Math.max(...allLevels.map(([, v]) => v), 1);

    for (let y = 0; y < heatmapSize; y++) {
      const priceLevelIndex = Math.floor((y / heatmapSize) * allLevels.length);
      const [, volume] = allLevels[priceLevelIndex] || [0, 0];
      
      // Normalize volume to 0-1 range, apply gamma for visual punch
      const normalizedVolume = Math.pow(volume / maxVolume, 0.5);
      newData[y * heatmapSize + heatmapSize - 1] = normalizedVolume;
    }

    state.heatmapData = newData;
  }, []);

  /**
   * Render loop - runs at 60fps independent of React
   */
  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    const state = renderState.current;

    if (!gl || !program || !texture) return;

    // Update time
    state.time += 0.016; // ~60fps

    // Update texture with new data
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      256,
      256,
      gl.RED,
      gl.FLOAT,
      state.heatmapData
    );

    // Clear
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use program
    gl.useProgram(program);

    // Set uniforms
    const timeLoc = gl.getUniformLocation(program, 'u_time');
    const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
    const textureLoc = gl.getUniformLocation(program, 'u_heatmapTexture');
    const midPriceLoc = gl.getUniformLocation(program, 'u_midPrice');
    const priceRangeLoc = gl.getUniformLocation(program, 'u_priceRange');
    const colorSchemeLoc = gl.getUniformLocation(program, 'u_colorScheme');

    gl.uniform1f(timeLoc, state.time);
    gl.uniform2f(resolutionLoc, gl.canvas.width, gl.canvas.height);
    gl.uniform1i(textureLoc, 0);
    gl.uniform1f(midPriceLoc, state.midPrice);
    gl.uniform1f(priceRangeLoc, state.priceRange);
    gl.uniform1i(colorSchemeLoc, state.colorScheme);

    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    animationFrameRef.current = requestAnimationFrame(render);
  }, []);

  // Register callback with WebSocket hook
  useEffect(() => {
    // This would be called by parent component with the socket callback
    // For now, we set up a window event listener
    const handleDataUpdate = (event: CustomEvent) => {
      updateHeatmapData(event.detail);
    };

    window.addEventListener('heatmap-data' as any, handleDataUpdate as any);
    return () => {
      window.removeEventListener('heatmap-data' as any, handleDataUpdate as any);
    };
  }, [updateHeatmapData]);

  // Initialize WebGL on mount
  useEffect(() => {
    initWebGL();
    
    // Start render loop
    animationFrameRef.current = requestAnimationFrame(render);

    // Handle resize
    const handleResize = () => {
      const canvas = canvasRef.current;
      const gl = glRef.current;
      if (!canvas || !gl) return;

      const displayWidth = canvas.clientWidth;
      const displayHeight = canvas.clientHeight;

      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
      
      // Cleanup WebGL
      const gl = glRef.current;
      if (gl) {
        gl.deleteProgram(programRef.current);
        gl.deleteTexture(textureRef.current);
      }
    };
  }, [initWebGL, render]);

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ 
          imageRendering: 'pixelated',
        }}
      />
      
      {/* Overlay labels */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 text-xs font-mono text-neon-cyan/50 writing-vertical">
        ASKS
      </div>
      <div className="absolute left-2 bottom-2 text-xs font-mono text-neon-magenta/50">
        BIDS
      </div>
      <div className="absolute right-2 top-2 text-xs font-mono text-gray-500">
        TIME →
      </div>
    </div>
  );
}
