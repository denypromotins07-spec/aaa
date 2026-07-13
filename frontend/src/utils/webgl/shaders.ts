/**
 * WebGL Shaders for Orderbook Heatmap
 * 
 * These shaders render the L2/L3 orderbook as a real-time heatmap:
 * - Y-axis: Price levels (bids below, asks above mid-price)
 * - X-axis: Time (scrolling from right to left)
 * - Color intensity: Volume/liquidity at each price level
 */

// Vertex Shader - Full screen quad
export const vertexShader = `#version 300 es
precision highp float;

in vec2 a_position;
in vec2 a_uv;

out vec2 v_uv;

void main() {
    v_uv = a_uv;
    
    // Convert from [-1, 1] clip space to screen coordinates
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Fragment Shader - Heatmap rendering with volume-based coloring
export const fragmentShader = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_heatmapTexture;
uniform float u_time;
uniform vec2 u_resolution;

// Cyberpunk color palette
vec3 colorLow = vec3(0.05, 0.05, 0.15);      // Deep blue for low liquidity
vec3 colorMid = vec3(0.0, 0.96, 1.0);         // Neon cyan for medium
vec3 colorHigh = vec3(1.0, 1.0, 0.0);         // Bright yellow for high
vec3 colorExtreme = vec3(1.0, 0.0, 1.0);      // Magenta for extreme walls

// Smooth color interpolation based on volume intensity
vec3 getColor(float intensity) {
    // Apply gamma correction for better visual distribution
    float gammaIntensity = pow(intensity, 0.7);
    
    if (gammaIntensity < 0.25) {
        return mix(colorLow, colorMid, gammaIntensity * 4.0);
    } else if (gammaIntensity < 0.5) {
        return mix(colorMid, colorHigh, (gammaIntensity - 0.25) * 4.0);
    } else if (gammaIntensity < 0.75) {
        return mix(colorHigh, colorExtreme, (gammaIntensity - 0.5) * 4.0);
    } else {
        return mix(colorExtreme, vec3(1.0), (gammaIntensity - 0.75) * 4.0);
    }
}

// Add scanline effect for cyberpunk aesthetic
float scanline(vec2 uv, float time) {
    float scanlinePos = mod(time * 0.5, 1.0);
    float scanlineThickness = 0.002;
    float scanline = smoothstep(scanlinePos - scanlineThickness, scanlinePos, uv.y);
    scanline *= smoothstep(scanlinePos + scanlineThickness, scanlinePos, uv.y);
    return 0.1 + scanline * 0.1;
}

// Vignette effect
float vignette(vec2 uv) {
    vec2 center = vec2(0.5);
    float dist = distance(uv, center);
    return 1.0 - smoothstep(0.3, 0.8, dist);
}

void main() {
    // Sample heatmap texture
    vec4 texel = texture(u_heatmapTexture, v_uv);
    
    // Get intensity from red channel (we encode volume there)
    float intensity = texel.r;
    
    // Apply color mapping
    vec3 color = getColor(intensity);
    
    // Add subtle grid lines
    float gridSize = 20.0;
    float gridX = step(0.98, fract(v_uv.x * gridSize));
    float gridY = step(0.98, fract(v_uv.y * gridSize));
    float grid = max(gridX, gridY) * 0.15;
    
    // Apply scanlines
    float scan = scanline(v_uv, u_time);
    
    // Apply vignette
    float vig = vignette(v_uv);
    
    // Combine all effects
    vec3 finalColor = color * (1.0 + grid) * scan * vig;
    
    // Add alpha fade at edges
    float alpha = smoothstep(0.0, 0.1, v_uv.x) * smoothstep(1.0, 0.9, v_uv.x);
    
    fragColor = vec4(finalColor, alpha);
}
`;

// Particle shader for trade visualization (Micro-Price Tape)
export const particleVertexShader = `#version 300 es
precision highp float;

in vec2 a_position;
in vec3 a_color;
in float a_size;
in float a_alpha;

out vec3 v_color;
out float v_alpha;

uniform vec2 u_resolution;

void main() {
    v_color = a_color;
    v_alpha = a_alpha;
    
    // Convert to clip space
    vec2 clipPos = (a_position / u_resolution) * 2.0 - 1.0;
    
    // Flip Y axis
    clipPos.y = -clipPos.y;
    
    gl_Position = vec4(clipPos, 0.0, 1.0);
    gl_PointSize = a_size;
}
`;

export const particleFragmentShader = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    // Circular particle with glow
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Soft edge
    float alpha = smoothstep(0.5, 0.3, dist) * v_alpha;
    
    // Inner glow
    float glow = 1.0 - dist * 2.0;
    glow = pow(glow, 1.5);
    
    fragColor = vec4(v_color * (1.0 + glow * 0.5), alpha);
}
`;

// Utility function to create shader program
export function createShaderProgram(
  gl: WebGL2RenderingContext,
  vsSource: string,
  fsSource: string
): WebGLProgram | null {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  if (!vs) return null;
  
  gl.shaderSource(vs, vsSource);
  gl.compileShader(vs);
  
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    console.error('Vertex shader compile error:', gl.getShaderInfoLog(vs));
    gl.deleteShader(vs);
    return null;
  }
  
  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  if (!fs) {
    gl.deleteShader(vs);
    return null;
  }
  
  gl.shaderSource(fs, fsSource);
  gl.compileShader(fs);
  
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    console.error('Fragment shader compile error:', gl.getShaderInfoLog(fs));
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  
  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    return null;
  }
  
  // Clean up shaders (they're linked into the program now)
  gl.detachShader(program, vs);
  gl.detachShader(program, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  
  return program;
}
