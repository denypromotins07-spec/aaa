'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { useNexusStore, TradeTick } from '@/store/nexusStore';
import { 
  createProgram, 
  PARTICLE_VERTEX_SHADER, 
  PARTICLE_FRAGMENT_SHADER 
} from '@/utils/webgl/shaders';

interface MicroPriceTapeProps {
  width?: number;
  height?: number;
  className?: string;
}

/**
 * Micro-Price Tape with Particle Effects
 * 
 * Renders executed trades as floating particles shooting across the screen:
 * - Green particles: Aggressive buys
 * - Red particles: Aggressive sells
 * - Particles leave trails indicating momentum
 * 
 * CRITICAL: Uses WebGL for particle rendering at 60fps, decoupled from React
 */
export function MicroPriceTape({ 
  width = 800, 
  height = 200,
  className = ''
}: MicroPriceTapeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Particle system state (outside React)
  const particleState = useRef({
    particles: [] as Particle[],
    maxParticles: 500,
    time: 0,
  });

  interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
    life: number;
    maxLife: number;
    isBuy: boolean;
    price: number;
    volume: number;
  }

  /**
   * Add a new trade particle
   */
  const addParticle = useCallback((trade: TradeTick) => {
    const state = particleState.current;
    
    // Create particle at right edge
    const particle: Particle = {
      x: 1.0, // Normalized device coords (right edge)
      y: 0.5 + (Math.random() - 0.5) * 0.3, // Random vertical position
      vx: -0.002 - Math.random() * 0.002, // Move left
      vy: (Math.random() - 0.5) * 0.001, // Slight vertical drift
      size: 4 + Math.random() * 4,
      life: 1.0,
      maxLife: 2.0 + Math.random(),
      isBuy: trade.isBuy,
      price: trade.price,
      volume: trade.volume,
    };
    
    state.particles.push(particle);
    
    // Trim if over limit
    if (state.particles.length > state.maxParticles) {
      state.particles.shift();
    }
  }, []);

  /**
   * Initialize WebGL
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

    const program = createProgram(gl, PARTICLE_VERTEX_SHADER, PARTICLE_FRAGMENT_SHADER);
    if (!program) {
      console.error('Failed to create particle shader program');
      return;
    }
    programRef.current = program;

    gl.clearColor(0.04, 0.04, 0.05, 0.0); // Transparent background
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }, []);

  /**
   * Update and render particles
   */
  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const state = particleState.current;

    if (!gl || !program) return;

    state.time += 0.016;

    // Update particles
    for (let i = state.particles.length - 1; i >= 0; i--) {
      const p = state.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= 0.016;
      
      // Remove dead or off-screen particles
      if (p.life <= 0 || p.x < -0.1) {
        state.particles.splice(i, 1);
      }
    }

    // Clear
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Render particles
    gl.useProgram(program);

    // Get locations
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const colorLoc = gl.getAttribLocation(program, 'a_color');
    const sizeLoc = gl.getAttribLocation(program, 'a_size');
    const alphaLoc = gl.getAttribLocation(program, 'a_alpha');
    const transformLoc = gl.getUniformLocation(program, 'u_transform');

    // Create particle data arrays
    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];
    const alphas: number[] = [];

    state.particles.forEach((p) => {
      positions.push(p.x, p.y);
      
      // Color based on buy/sell
      if (p.isBuy) {
        colors.push(0.0, 1.0, 0.6); // Neon green
      } else {
        colors.push(1.0, 0.2, 0.4); // Neon red
      }
      
      sizes.push(p.size * p.life); // Shrink as they die
      alphas.push(p.life);
    });

    // Upload data
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    // Interleaved vertex data
    const vertexData = new Float32Array(
      positions.length + colors.length + sizes.length + alphas.length
    );
    
    let offset = 0;
    vertexData.set(positions, offset);
    offset += positions.length;
    vertexData.set(colors, offset);
    offset += colors.length;
    vertexData.set(sizes, offset);
    offset += sizes.length;
    vertexData.set(alphas, offset);

    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);

    // Set up attributes
    const stride = 7 * Float32Array.BYTES_PER_ELEMENT;
    
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, stride, 0);

    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);

    gl.enableVertexAttribArray(sizeLoc);
    gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, stride, 5 * Float32Array.BYTES_PER_ELEMENT);

    gl.enableVertexAttribArray(alphaLoc);
    gl.vertexAttribPointer(alphaLoc, 1, gl.FLOAT, false, stride, 6 * Float32Array.BYTES_PER_ELEMENT);

    // Identity transform (could add perspective effects)
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    gl.uniformMatrix3fv(transformLoc, false, identity);

    // Draw
    gl.drawArrays(gl.POINTS, 0, state.particles.length);

    gl.deleteVertexArray(vao);
    gl.deleteBuffer(vbo);

    animationFrameRef.current = requestAnimationFrame(render);
  }, []);

  // Subscribe to trade updates from store
  useEffect(() => {
    const unsubscribe = useNexusStore.subscribe(
      (state) => state.recentTrades,
      (trades, previousTrades) => {
        // Only process new trades
        if (previousTrades && trades.length > previousTrades.length) {
          const newTrades = trades.slice(previousTrades.length);
          newTrades.forEach((trade) => addParticle(trade));
        }
      }
    );

    return () => {
      unsubscribe();
    };
  }, [addParticle]);

  // Initialize WebGL on mount
  useEffect(() => {
    initWebGL();
    animationFrameRef.current = requestAnimationFrame(render);

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
      
      const gl = glRef.current;
      if (gl) {
        gl.deleteProgram(programRef.current);
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
      
      {/* Overlay info */}
      <div className="absolute left-4 top-2 text-xs font-mono text-gray-500">
        TRADE FLOW
      </div>
      <div className="absolute right-4 top-2 flex gap-4 text-xs font-mono">
        <span className="text-neon-green">● BUY</span>
        <span className="text-neon-red">● SELL</span>
      </div>
    </div>
  );
}
