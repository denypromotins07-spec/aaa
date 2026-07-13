// Shared WebGL shaders for Omega Point visualizations

export const omegaVertexShader = `
  uniform float uTime;
  uniform float uCollapseFactor;
  attribute float aIndex;
  
  varying vec3 vColor;
  varying float vAlpha;
  varying vec3 vPosition;
  
  void main() {
    float t = uTime * 0.1;
    float idx = aIndex * 0.01;
    
    // Phase space trajectory
    float x = sin(t + idx * 0.5) * cos(idx * 0.2);
    float y = cos(t * 0.7 + idx * 0.3) * sin(idx * 0.4);
    float z = sin(t * 0.4 + idx * 0.6) * 0.5;
    
    // Apply collapse factor (Omega Point approach)
    float collapse = 1.0 - uCollapseFactor * 0.7;
    
    vec3 newPos = position;
    newPos.x += x * collapse * 3.0;
    newPos.y += y * collapse * 3.0;
    newPos.z += z * collapse * 3.0;
    
    vPosition = newPos;
    
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Color shifts as we approach Omega Point
    vec3 earlyColor = vec3(0.0, 1.0, 1.0);  // Cyan
    vec3 lateColor = vec3(1.0, 0.0, 1.0);   // Magenta
    vColor = mix(earlyColor, lateColor, uCollapseFactor);
    vAlpha = 0.6 + 0.4 * sin(t + idx);
  }
`;

export const omegaFragmentShader = `
  varying vec3 vColor;
  varying float vAlpha;
  varying vec3 vPosition;
  
  void main() {
    if (vAlpha < 0.1) discard;
    
    // Add glow based on position
    float dist = length(vPosition);
    float glow = exp(-dist * 0.5);
    
    vec3 finalColor = vColor * (1.0 + glow);
    gl_FragColor = vec4(finalColor, vAlpha);
  }
`;

export const thermodynamicSphereVertexShader = `
  uniform float uCollapseFactor;
  uniform float uTime;
  
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vHologram;
  
  void main() {
    vNormal = normal;
    
    // Collapse sphere radius
    float scale = 1.0 - uCollapseFactor * 0.85;
    vPosition = position * scale;
    
    // Holographic grid animation
    float grid = sin(vPosition.x * 15.0 + uTime) * 
                 sin(vPosition.y * 15.0 + uTime * 0.7) * 
                 sin(vPosition.z * 15.0 + uTime * 0.5);
    vHologram = smoothstep(-0.2, 0.2, grid);
    
    vec4 mvPosition = modelViewMatrix * vec4(vPosition, 1.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const thermodynamicSphereFragmentShader = `
  uniform float uCollapseFactor;
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vHologram;
  
  void main() {
    // Color gradient from cyan (healthy) to red (exhausted)
    vec3 healthyColor = vec3(0.0, 1.0, 1.0);
    vec3 exhaustedColor = vec3(1.0, 0.2, 0.2);
    vec3 baseColor = mix(healthyColor, exhaustedColor, uCollapseFactor);
    
    // Holographic transparency pattern
    float alpha = 0.2 + 0.5 * vHologram;
    
    // Fresnel-like edge highlighting
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.0);
    
    vec3 finalColor = baseColor + fresnel * 0.3;
    gl_FragColor = vec4(finalColor, alpha);
  }
`;

// Utility function for calculating thermodynamic efficiency
export function calculateLandauerEfficiency(energyDissipation: number, temperature: number): number {
  // Landauer's principle: minimum energy to erase one bit
  const k_B = 1.380649e-23; // Boltzmann constant
  const landauerLimit = k_B * temperature * Math.log(2);
  
  // Efficiency ratio (actual vs theoretical minimum)
  return Math.min(1.0, landauerLimit / energyDissipation);
}

// Poincaré recurrence time estimation (simplified)
export function estimatePoincareRecurrence(phaseSpaceVolume: number, precision: number): number {
  // Extremely simplified formula - real calculation would be much more complex
  return Math.exp(phaseSpaceVolume * precision);
}
