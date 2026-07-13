'use client';

import React, { useEffect, useRef, useState } from 'react';
import { THERMAL_VERTEX_SHADER, THERMAL_FRAGMENT_SHADER } from '../../utils/webgl/thermalShaders';

interface SiliconThermalHeatmapProps {
  width: number;
  height: number;
  minTemp?: number;
  maxTemp?: number;
}

const GRID_SIZE = 64;

export default function SiliconThermalHeatmap({ 
  width, 
  height, 
  minTemp = 30, 
  maxTemp = 95 
}: SiliconThermalHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  const thermalDataRef = useRef<Float32Array>(new Float32Array(GRID_SIZE * GRID_SIZE));
  const [contextLost, setContextLost] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl2', { 
      antialias: false, 
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    });
    
    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    glRef.current = gl;

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, THERMAL_VERTEX_SHADER);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, THERMAL_FRAGMENT_SHADER);
    
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

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    const dummyData = new Float32Array(GRID_SIZE * GRID_SIZE).fill(50);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, GRID_SIZE, GRID_SIZE, 0, gl.RED, gl.FLOAT, dummyData);
    
    textureRef.current = texture;

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1
    ]), gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const handleContextLost = (e: WebGLContextEvent) => {
      e.preventDefault();
      setContextLost(true);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };

    const handleContextRestored = () => {
      setContextLost(false);
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    };

    canvas.addEventListener('webglcontextlost', handleContextLost, false);
    canvas.addEventListener('webglcontextrestored', handleContextRestored, false);

    return () => {
      canvas.removeEventListener('webglcontextlost', handleContextLost);
      canvas.removeEventListener('webglcontextrestored', handleContextRestored);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (texture) gl.deleteTexture(texture);
      if (program) gl.deleteProgram(program);
      if (vertexShader) gl.deleteShader(vertexShader);
      if (fragmentShader) gl.deleteShader(fragmentShader);
    };
  }, []);

  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    
    if (!gl || !program || !texture || contextLost) return;

    const render = () => {
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.04, 0.04, 0.05, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, gl.RED, gl.FLOAT, thermalDataRef.current);

      const minTempLoc = gl.getUniformLocation(program, 'u_minTemp');
      const maxTempLoc = gl.getUniformLocation(program, 'u_maxTemp');
      gl.uniform1f(minTempLoc, minTemp);
      gl.uniform1f(maxTempLoc, maxTemp);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      
      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height, minTemp, maxTemp, contextLost]);

  useEffect(() => {
    const handleDataUpdate = (event: CustomEvent<Float32Array>) => {
      if (event.detail && event.detail.length === GRID_SIZE * GRID_SIZE) {
        thermalDataRef.current = event.detail;
      }
    };

    window.addEventListener('thermal-data-update' as any, handleDataUpdate as any);
    return () => {
      window.removeEventListener('thermal-data-update' as any, handleDataUpdate as any);
    };
  }, []);

  return (
    <div className="relative w-full h-full glass-panel rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-2 left-3 z-10 font-mono text-xs text-cyan-400 tracking-wider">
        SILICON_THERMAL_MAP :: FPGA_ARRAY_01
      </div>
      <div className="absolute bottom-2 right-3 z-10 font-mono text-xs text-pink-400">
        MIN: {minTemp}°C | MAX: {maxTemp}°C
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
      {contextLost && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="text-red-500 font-mono animate-pulse">CONTEXT_LOST :: REINITIALIZING...</div>
        </div>
      )}
    </div>
  );
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
