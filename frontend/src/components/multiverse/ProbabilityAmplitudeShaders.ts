// GLSL shaders for probability amplitude visualization

export const probabilityVertexShader = `
  uniform float uTime;
  attribute float uUtility;
  attribute float uIsSurvival;
  
  varying vec3 vColor;
  varying float vAlpha;
  varying float vUtility;
  
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Store utility for fragment shader
    vUtility = uUtility;
    
    // Color mapping based on utility
    vec3 profitColor = vec3(0.0, 1.0, 1.0);    // Cyan for profit
    vec3 neutralColor = vec3(0.5, 0.5, 0.8);   // Purple-gray for neutral
    vec3 ruinColor = vec3(0.05, 0.05, 0.1);    // Near-black for ruin
    vec3 survivalColor = vec3(1.0, 1.0, 1.0);  // White for survival branch
    
    if (uIsSurvival > 0.5) {
      vColor = survivalColor;
      vAlpha = 0.9 + 0.1 * sin(uTime * 5.0);
    } else {
      float normalizedUtility = (uUtility + 1.0) * 0.5;
      
      if (normalizedUtility > 0.6) {
        vColor = mix(neutralColor, profitColor, (normalizedUtility - 0.6) * 2.5);
        vAlpha = 0.4 + normalizedUtility * 0.4;
      } else if (normalizedUtility > 0.3) {
        vColor = neutralColor;
        vAlpha = 0.3 + normalizedUtility * 0.3;
      } else {
        vColor = mix(ruinColor, neutralColor, normalizedUtility * 3.33);
        vAlpha = 0.2 + normalizedUtility * 0.2;
      }
    }
    
    // Size attenuation based on depth/distance
    float dist = length(position);
    gl_PointSize = (20.0 / dist) * (1.0 + uUtility * 0.5);
    gl_PointSize = clamp(gl_PointSize, 1.0, 8.0);
  }
`;

export const probabilityFragmentShader = `
  varying vec3 vColor;
  varying float vAlpha;
  varying float vUtility;
  
  void main() {
    if (vAlpha < 0.05) discard;
    
    // Circular particle shape
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) discard;
    
    // Add glow effect
    float glow = 1.0 - length(coord) * 2.0;
    glow = pow(glow, 1.5);
    
    vec3 finalColor = vColor * (1.0 + glow * 0.5);
    gl_FragColor = vec4(finalColor, vAlpha);
  }
`;

export const bloomPostProcessFragment = `
  uniform sampler2D uTexture;
  uniform float uIntensity;
  uniform float uThreshold;
  
  varying vec2 vUv;
  
  void main() {
    vec4 color = texture2D(uTexture, vUv);
    
    // Extract bright areas
    float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float bloom = smoothstep(uThreshold, 1.0, brightness) * uIntensity;
    
    // Simple blur approximation (in production, use multi-pass Gaussian)
    vec4 blurred = vec4(0.0);
    float totalWeight = 0.0;
    
    for (float x = -2.0; x <= 2.0; x++) {
      for (float y = -2.0; y <= 2.0; y++) {
        vec2 offset = vec2(x, y) * 0.002;
        vec4 sampleColor = texture2D(uTexture, vUv + offset);
        float weight = exp(-0.5 * (x*x + y*y) / 2.0);
        blurred += sampleColor * weight;
        totalWeight += weight;
      }
    }
    blurred /= totalWeight;
    
    // Combine original with bloom
    vec3 result = color.rgb + blurred.rgb * bloom;
    gl_FragColor = vec4(result, color.a);
  }
`;

// Utility functions for calculating financial utility from PnL data
export function calculateUtility(pnl: number, riskAdjustment: number = 1.0): number {
  // Normalize PnL to [-1, 1] range using sigmoid-like function
  const rawUtility = Math.tanh(pnl * riskAdjustment);
  return Math.max(-1, Math.min(1, rawUtility));
}

export function identifySurvivalBranch(branches: Array<{ id: number; utility: number }>): number {
  // Find the branch with highest utility that avoids catastrophic loss
  const viableBranches = branches.filter(b => b.utility > -0.8);
  if (viableBranches.length === 0) {
    return branches.reduce((max, b) => (b.utility > max.utility ? b : max), branches[0]).id;
  }
  return viableBranches.reduce((max, b) => (b.utility > max.utility ? b : max), viableBranches[0]).id;
}
