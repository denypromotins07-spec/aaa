export const RADAR_VERTEX_SHADER = `#version 300 es
in vec2 position;
uniform float uTime;
uniform float uPulse;

void main() {
    vec2 pos = position * (1.0 + uPulse * 0.05 * sin(uTime * 2.0));
    gl_Position = vec4(pos, 0.0, 1.0);
}
`;

export const RADAR_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec3 uColor;
uniform float uGlow;
uniform int uAxisCount;

out vec4 fragColor;

float gridPattern(vec2 uv, int divisions) {
    vec2 f = fract(uv * float(divisions));
    return step(0.98, f.x) + step(0.98, f.y);
}

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(800.0, 600.0);
    vec2 center = vec2(0.5);
    float dist = length(uv - center);
    
    // Concentric rings
    float rings = sin(dist * 20.0 + uTime) * 0.5 + 0.5;
    rings *= smoothstep(0.5, 0.0, dist);
    
    // Axis lines
    float axes = 0.0;
    for(int i = 0; i < 6; i++) {
        float angle = float(i) * 3.14159 / 3.0;
        vec2 dir = vec2(cos(angle), sin(angle));
        float lineDist = abs(dot(normalize(uv - center), normalize(vec2(-dir.y, dir.x))));
        axes += smoothstep(0.02, 0.0, lineDist) * smoothstep(0.5, 0.0, dist);
    }
    
    float intensity = rings + axes + uGlow * 0.5;
    vec3 finalColor = uColor * intensity;
    
    // Add holographic scanline effect
    float scanline = sin(gl_FragCoord.y * 0.1 + uTime * 5.0) * 0.1 + 0.9;
    finalColor *= scanline;
    
    fragColor = vec4(finalColor, 0.8);
}
`;

export const ORBIT_VERTEX_SHADER = `#version 300 es
in vec3 position;
in vec3 color;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec3 vColor;

void main() {
    vColor = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const ORBIT_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec3 vColor;
uniform float uAlpha;

out vec4 fragColor;

void main() {
    fragColor = vec4(vColor, uAlpha);
}
`;
