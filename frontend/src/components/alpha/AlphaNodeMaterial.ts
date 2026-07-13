import * as THREE from 'three';

/**
 * NEXUS-OMEGA FRONTEND STAGE 2
 * Module: Alpha Node Material Shader
 * Purpose: Custom GLSL shader for alpha nodes with pulsing emission based on conviction.
 */

export const vertexShader = `
  varying vec3 vPosition;
  varying vec2 vUv;
  uniform float uTime;
  uniform float uConviction;
  
  void main() {
    vUv = uv;
    vPosition = position;
    
    // Pulse effect based on conviction and time
    float pulse = sin(uTime * 5.0 + uConviction * 10.0) * 0.1 * uConviction;
    vec3 newPosition = position * (1.0 + pulse);
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
    gl_PointSize = 8.0 * (1.0 + pulse);
  }
`;

export const fragmentShader = `
  varying vec3 vPosition;
  varying vec2 vUv;
  uniform vec3 uColor;
  uniform float uConviction;
  uniform float uTime;
  
  void main() {
    // Circular node shape
    float dist = length(vUv - 0.5);
    if (dist > 0.5) discard;
    
    // Gradient from center to edge
    float gradient = 1.0 - (dist * 2.0);
    
    // Pulsing glow ring
    float glow = sin(uTime * 3.0) * 0.5 + 0.5;
    glow = pow(glow, 3.0) * uConviction;
    
    vec3 finalColor = uColor * gradient + vec3(glow);
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

export function createAlphaNodeMaterial(conviction: number, color: [number, number, number]): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uConviction: { value: conviction },
      uColor: { value: new THREE.Color().setRGB(color[0] / 255, color[1] / 255, color[2] / 255) }
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });
}

/**
 * Updates the material uniforms for animation.
 * Mutates in place to avoid allocations.
 */
export function updateAlphaNodeMaterial(
  material: THREE.ShaderMaterial,
  conviction: number,
  color: [number, number, number]
): void {
  material.uniforms.uConviction.value = conviction;
  material.uniforms.uColor.value.setRGB(color[0] / 255, color[1] / 255, color[2] / 255);
}
