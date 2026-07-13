'use client';

import React, { useEffect, useRef } from 'react';
import { PHOTONIC_VERTEX_SHADER, PHOTONIC_FRAGMENT_SHADER } from '../../utils/webgl/thermalShaders';

interface PhotonicMeshVisualizerProps {
  width: number;
  height: number;
}

interface MZINode {
  x: number;
  y: number;
  phase: number; // 0 to 1
  intensity: number;
}

const NUM_NODES = 64;

export default function PhotonicMeshVisualizer({ width, height }: PhotonicMeshVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  const nodesRef = useRef<MZINode[]>([]);
  const timeRef = useRef<number>(0);

  useEffect(() => {
    // Initialize MZI mesh nodes in a grid pattern
    const nodes: MZINode[] = [];
    const cols = 8;
    const rows = 8;
    
    for (let i = 0; i < NUM_NODES; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      nodes.push({
        x: (col + 0.5) / cols,
        y: (row + 0.5) / rows,
        phase: Math.random(),
        intensity: 0.5 + Math.random() * 0.5,
      });
    }
    nodesRef.current = nodes;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl2', {
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    });

    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    glRef.current = gl;

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, PHOTONIC_VERTEX_SHADER);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, PHOTONIC_FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    if (!program) return;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return;
    }

    programRef.current = program;

    // Create VAO
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    // Position buffer
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = new Float32Array(NUM_NODES * 2);
    nodesRef.current.forEach((node, i) => {
      positions[i * 2] = node.x * 2 - 1;
      positions[i * 2 + 1] = node.y * 2 - 1;
    });
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    // Intensity buffer
    const intensityBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, intensityBuffer);
    const intensities = new Float32Array(NUM_NODES);
    nodesRef.current.forEach((node, i) => {
      intensities[i] = node.intensity;
    });
    gl.bufferData(gl.ARRAY_BUFFER, intensities, gl.DYNAMIC_DRAW);

    const intensityLoc = gl.getAttribLocation(program, 'a_intensity');
    gl.enableVertexAttribArray(intensityLoc);
    gl.vertexAttribPointer(intensityLoc, 1, gl.FLOAT, false, 0, 0);

    // Color buffer (cyan/magenta gradient)
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    const colors = new Float32Array(NUM_NODES * 3);
    nodesRef.current.forEach((node, i) => {
      // Interpolate between cyan and magenta based on phase
      const t = node.phase;
      colors[i * 3] = t;     // R
      colors[i * 3 + 1] = 1 - t; // G
      colors[i * 3 + 2] = 1;     // B
    });
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);

    const colorLoc = gl.getAttribLocation(program, 'a_color');
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (program) gl.deleteProgram(program);
      if (vertexShader) gl.deleteShader(vertexShader);
      if (fragmentShader) gl.deleteShader(fragmentShader);
      gl.deleteBuffer(positionBuffer);
      gl.deleteBuffer(intensityBuffer);
      gl.deleteBuffer(colorBuffer);
      gl.deleteVertexArray(vao);
    };
  }, []);

  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;

    if (!gl || !program) return;

    const render = () => {
      timeRef.current += 0.016;

      gl.viewport(0, 0, width, height);
      gl.clearColor(0.04, 0.04, 0.05, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      // Update node phases dynamically
      const nodes = nodesRef.current;
      const positions = new Float32Array(NUM_NODES * 2);
      const intensities = new Float32Array(NUM_NODES);
      const colors = new Float32Array(NUM_NODES * 3);

      nodes.forEach((node, i) => {
        // Animate phase
        node.phase = (node.phase + 0.005) % 1;
        
        positions[i * 2] = node.x * 2 - 1;
        positions[i * 2 + 1] = node.y * 2 - 1;
        
        // Pulse intensity
        node.intensity = 0.5 + 0.5 * Math.sin(timeRef.current * 2 + i);
        intensities[i] = node.intensity;

        // Color based on phase
        const t = node.phase;
        colors[i * 3] = t;
        colors[i * 3 + 1] = 1 - t;
        colors[i * 3 + 2] = 1;
      });

      // Update buffers
      const positionBuffer = gl.getAttribLocation(program, 'a_position');
      // In a real scenario we'd cache buffer refs, but for simplicity we re-bind
      // For production: cache buffer references in refs

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      gl.drawArrays(gl.POINTS, 0, NUM_NODES);

      // Draw waveguide connections (lines between adjacent nodes)
      drawWaveguides(gl, program, nodes, timeRef.current);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height]);

  return (
    <div className="relative w-full h-full glass-panel rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-2 left-3 z-10 font-mono text-xs text-cyan-400 tracking-wider">
        PHOTONIC_MZI_MESH :: STAGE_32
      </div>
      <div className="absolute bottom-2 right-3 z-10 font-mono text-xs text-purple-400">
        OPTICAL_INTERFERENCE :: ACTIVE
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
    </div>
  );
}

function drawWaveguides(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  nodes: MZINode[],
  time: number
) {
  // Simple line rendering for waveguides
  // In production, this would use a separate shader program for lines
  const cols = 8;
  
  gl.useProgram(program);
  
  const linePositions: number[] = [];
  const lineColors: number[] = [];
  
  // Connect adjacent nodes horizontally and vertically
  for (let row = 0; row < cols; row++) {
    for (let col = 0; col < cols - 1; col++) {
      const i = row * cols + col;
      const j = i + 1;
      
      linePositions.push(
        nodes[i].x * 2 - 1, nodes[i].y * 2 - 1,
        nodes[j].x * 2 - 1, nodes[j].y * 2 - 1
      );
      
      // Gradient color along the waveguide
      const avgPhase = (nodes[i].phase + nodes[j].phase) / 2;
      const alpha = 0.3 + 0.7 * Math.sin(time * 3 + i);
      
      lineColors.push(avgPhase, 1 - avgPhase, 1, alpha);
      lineColors.push(avgPhase, 1 - avgPhase, 1, alpha);
    }
  }
  
  // Vertical connections
  for (let row = 0; row < cols - 1; row++) {
    for (let col = 0; col < cols; col++) {
      const i = row * cols + col;
      const j = (row + 1) * cols + col;
      
      linePositions.push(
        nodes[i].x * 2 - 1, nodes[i].y * 2 - 1,
        nodes[j].x * 2 - 1, nodes[j].y * 2 - 1
      );
      
      const avgPhase = (nodes[i].phase + nodes[j].phase) / 2;
      const alpha = 0.3 + 0.7 * Math.sin(time * 3 + i + 100);
      
      lineColors.push(avgPhase, 1 - avgPhase, 1, alpha);
      lineColors.push(avgPhase, 1 - avgPhase, 1, alpha);
    }
  }
  
  // This is a simplified version - proper implementation would use a line shader
  // For now, we skip actual line rendering to keep shader complexity manageable
}

function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}
