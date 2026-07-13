/**
 * WebGL Shaders for Orderbook Heatmap
 * 
 * These shaders render the L2/L3 orderbook as a real-time heatmap:
 * - Y-axis: Price levels (bids below, asks above)
 * - X-axis: Time (scrolling right to left)
 * - Color: Volume/liquidity intensity
 */

// Vertex Shader - Full screen quad
export const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in vec2 a_uv;
out vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Fragment Shader - Heatmap rendering with cyberpunk aesthetic
export const HEATMAP_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform sampler2D u_heatmapTexture;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_midPrice;
uniform float u_priceRange;
uniform int u_colorScheme;

in vec2 v_uv;
out vec4 fragColor;

// Cyberpunk color palette
vec3 getColorCyberpunk(float intensity) {
    // Deep blue for low intensity
    vec3 colorLow = vec3(0.0, 0.1, 0.3);
    // Cyan for medium
    vec3 colorMid = vec3(0.0, 0.96, 1.0);
    // Magenta for high
    vec3 colorHigh = vec3(1.0, 0.0, 1.0);
    // Bright yellow for extreme
    vec3 colorExtreme = vec3(1.0, 1.0, 0.0);
    
    if (intensity < 0.33) {
        return mix(colorLow, colorMid, intensity / 0.33);
    } else if (intensity < 0.66) {
        return mix(colorMid, colorHigh, (intensity - 0.33) / 0.33);
    } else {
        return mix(colorHigh, colorExtreme, (intensity - 0.66) / 0.34);
    }
}

// Matrix color palette
vec3 getColorMatrix(float intensity) {
    vec3 colorLow = vec3(0.0, 0.05, 0.0);
    vec3 colorMid = vec3(0.0, 0.5, 0.0);
    vec3 colorHigh = vec3(0.0, 1.0, 0.0);
    vec3 colorExtreme = vec3(0.5, 1.0, 0.5);
    
    return mix(colorLow, colorMid, intensity * 1.5);
}

// Monochrome palette
vec3 getColorMono(float intensity) {
    return vec3(intensity);
}

// Scanline effect
float scanline(float y, float time) {
    float scan = sin(y * 3.14159 * 2.0 - time * 0.5);
    return 0.95 + 0.05 * scan;
}

// Grid overlay
float grid(vec2 uv, float gridSize) {
    vec2 gridUV = fract(uv * gridSize);
    float line = step(0.98, gridUV.x) + step(0.98, gridUV.y);
    return 1.0 - min(1.0, line * 0.3);
}

void main() {
    vec2 uv = v_uv;
    
    // Sample heatmap texture
    vec4 texColor = texture(u_heatmapTexture, uv);
    float intensity = texColor.r;
    
    // Apply color scheme
    vec3 color;
    if (u_colorScheme == 0) {
        color = getColorCyberpunk(intensity);
    } else if (u_colorScheme == 1) {
        color = getColorMatrix(intensity);
    } else {
        color = getColorMono(intensity);
    }
    
    // Add scanline effect
    float scan = scanline(uv.y, u_time);
    color *= scan;
    
    // Add subtle grid
    float gridOverlay = grid(uv, 20.0);
    color *= gridOverlay;
    
    // Highlight mid-price level
    float midY = 0.5;
    float distToMid = abs(uv.y - midY);
    float midHighlight = exp(-distToMid * 50.0) * 0.3;
    color += vec3(0.0, 0.5, 0.5) * midHighlight;
    
    // Vignette
    vec2 centeredUV = uv * 2.0 - 1.0;
    float vignette = 1.0 - dot(centeredUV, centeredUV) * 0.3;
    color *= vignette;
    
    // Alpha based on intensity (fade out empty areas)
    float alpha = max(0.1, intensity);
    
    fragColor = vec4(color, alpha);
}
`;

// Trade particle vertex shader
export const PARTICLE_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in vec3 a_color;
in float a_size;
in float a_alpha;

uniform mat3 u_transform;

out vec3 v_color;
out float v_alpha;

void main() {
    vec3 transformed = u_transform * vec3(a_position, 1.0);
    gl_Position = vec4(transformed, 1.0);
    gl_PointSize = a_size;
    v_color = a_color;
    v_alpha = a_alpha;
}
`;

// Trade particle fragment shader
export const PARTICLE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    // Circular particle with glow
    vec2 center = gl_PointCoord - 0.5;
    float dist = length(center);
    
    // Soft edge
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    alpha *= v_alpha;
    
    // Glow center
    float glow = exp(-dist * 4.0);
    vec3 color = v_color * (1.0 + glow * 0.5);
    
    fragColor = vec4(color, alpha);
}
`;

// Price tape text rendering (simplified, uses canvas 2D for actual text)
export const TAPE_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in vec2 a_uv;
out vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

export const TAPE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform sampler2D u_tapeTexture;
uniform vec4 u_colorBuy;
uniform vec4 u_colorSell;
uniform float u_isBuy;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 tex = texture(u_tapeTexture, v_uv);
    
    // Mix buy/sell colors based on trade direction
    vec4 color = mix(u_colorSell, u_colorBuy, u_isBuy);
    
    fragColor = vec4(color.rgb, tex.a * color.a);
}
`;

/**
 * Helper function to create WebGL program
 */
export function createProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram | null {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  
  if (!vertexShader || !fragmentShader) return null;
  
  const program = gl.createProgram();
  if (!program) return null;
  
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('WebGL program link error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  
  return program;
}

function createShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('WebGL shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  
  return shader;
}
