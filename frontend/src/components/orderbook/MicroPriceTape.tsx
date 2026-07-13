/**
 * Micro-Price Tape Component
 * 
 * Renders executed trades as floating particles with trails:
 * - Green particles for aggressive buys (taker buy)
 * - Red particles for aggressive sells (taker sell)
 * - Particle size proportional to trade volume
 * - Smooth animation at 60fps using requestAnimationFrame
 * 
 * CRITICAL: Completely decoupled from React render cycle.
 */

'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { createShaderProgram, particleVertexShader, particleFragmentShader } from '@/utils/webgl/shaders';

interface Trade {
  price: number;
  volume: number;
  side: 0 | 1; // 0 = buy, 1 = sell
  timestamp: number;
}

interface MicroPriceTapeProps {
  width?: number;
  height?: number;
  latestFrameRef: React.MutableRefObject<{
    trades: [number, number, number][];
  } | null>;
}

const MAX_PARTICLES = 500;
const PARTICLE_LIFETIME = 3000; // ms

export default function MicroPriceTape({
  width = 400,
  height = 200,
  latestFrameRef,
}: MicroPriceTapeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Particle system state
  const particlesRef = useRef<Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
    color: [number, number, number];
    alpha: number;
    age: number;
    maxAge: number;
  }>>([]);
  
  // Track processed trade indices to avoid duplicates
  const lastTradeCountRef = useRef(0);
  
  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: false,
    });
    
    if (!gl) {
      console.error('[TAPE] WebGL2 not supported');
      return;
    }
    
    glRef.current = gl;
    
    const program = createShaderProgram(gl, particleVertexShader, particleFragmentShader);
    if (!program) {
      console.error('[TAPE] Failed to create shader program');
      return;
    }
    programRef.current = program;
    
    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      gl.deleteProgram(program);
    };
  }, []);
  
  // Add new trades as particles
  const addTrades = useCallback((trades: [number, number, number][]) => {
    if (trades.length <= lastTradeCountRef.current) return;
    
    const newTrades = trades.slice(lastTradeCountRef.current);
    lastTradeCountRef.current = trades.length;
    
    const prevPriceRef = { value: newTrades.length > 0 ? newTrades[0][0] : 0 };
    
    newTrades.forEach(([price, volume, side]) => {
      if (particlesRef.current.length >= MAX_PARTICLES) {
        // Remove oldest particles
        particlesRef.current.shift();
      }
      
      // Calculate Y position based on price
      const priceChange = price - prevPriceRef.value;
      const normalizedPriceChange = Math.max(-1, Math.min(1, priceChange / (price * 0.001)));
      const y = (0.5 + normalizedPriceChange * 0.3) * height;
      
      prevPriceRef.value = price;
      
      // Size based on volume (log scale)
      const baseSize = Math.log10(volume + 1) * 3;
      const size = Math.min(baseSize, 20);
      
      // Color: green for buys, red for sells
      const color: [number, number, number] = side === 0 
        ? [0, 1, 0.6]  // Neon green
        : [1, 0.17, 0.43]; // Neon red
      
      particlesRef.current.push({
        x: 0, // Start from left
        y,
        vx: 50 + Math.random() * 30, // pixels per second
        vy: (Math.random() - 0.5) * 20,
        size,
        color,
        alpha: 1,
        age: 0,
        maxAge: PARTICLE_LIFETIME + Math.random() * 1000,
      });
    });
  }, [height]);
  
  // Animation loop
  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;
    
    if (!gl || !program) return;
    
    let lastTime = performance.now();
    
    const render = () => {
      const now = performance.now();
      const deltaTime = (now - lastTime) / 1000; // seconds
      lastTime = now;
      
      // Check for new trades
      const frame = latestFrameRef.current;
      if (frame?.trades && frame.trades.length > 0) {
        addTrades(frame.trades);
      }
      
      // Update particles
      particlesRef.current = particlesRef.current.filter((p) => {
        p.age += deltaTime * 1000;
        p.x += p.vx * deltaTime;
        p.y += p.vy * deltaTime;
        p.alpha = 1 - p.age / p.maxAge;
        
        return p.age < p.maxAge && p.x < width + 50;
      });
      
      // Clear canvas
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
      gl.clearColor(0.04, 0.04, 0.05, 0); // Transparent background
      gl.clear(gl.COLOR_BUFFER_BIT);
      
      // Enable blending for glow effect
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      
      gl.useProgram(program);
      
      // Set resolution uniform
      const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
      gl.uniform2f(resolutionLoc, gl.canvas.width, gl.canvas.height);
      
      // Render each particle
      particlesRef.current.forEach((p) => {
        // Create particle vertex data
        const halfSize = p.size / 2;
        const vertices = new Float32Array([
          p.x - halfSize, p.y - halfSize,
          p.x + halfSize, p.y - halfSize,
          p.x - halfSize, p.y + halfSize,
          p.x - halfSize, p.y + halfSize,
          p.x + halfSize, p.y - halfSize,
          p.x + halfSize, p.y + halfSize,
        ]);
        
        const colors = new Float32Array([
          ...p.color, p.alpha,
          ...p.color, p.alpha,
          ...p.color, p.alpha,
          ...p.color, p.alpha,
          ...p.color, p.alpha,
          ...p.color, p.alpha,
        ]);
        
        // Set up buffers (simplified - in production use VAOs)
        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        
        const positionLoc = gl.getAttribLocation(program, 'a_position');
        const colorLoc = gl.getAttribLocation(program, 'a_color');
        const sizeLoc = gl.getAttribLocation(program, 'a_size');
        const alphaLoc = gl.getAttribLocation(program, 'a_alpha');
        
        // Interleaved buffer would be more efficient
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
        
        // Draw point sprite
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        
        gl.deleteBuffer(vbo);
      });
      
      gl.disable(gl.BLEND);
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [latestFrameRef, addTrades, width, height]);
  
  return (
    <div className="relative w-full h-full overflow-hidden">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
      
      {/* Legend overlay */}
      <div className="absolute bottom-2 right-2 flex gap-3 text-xs font-mono">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-neon-green" />
          <span className="text-gray-400">BUY</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-neon-red" />
          <span className="text-gray-400">SELL</span>
        </div>
      </div>
    </div>
  );
}
