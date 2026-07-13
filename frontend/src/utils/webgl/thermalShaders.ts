// Thermal Gradient & Photonic Interference Shaders
// Optimized for WebGL2 Fragment Shaders

export const THERMAL_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

export const THERMAL_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform float u_minTemp;
uniform float u_maxTemp;
out vec4 fragColor;

vec3 thermalPalette(float t) {
    // Cyberpunk Thermal Palette: Deep Blue -> Cyan -> Magenta -> White
    t = clamp(t, 0.0, 1.0);
    vec3 color1 = vec3(0.0, 0.0, 0.1);   // Deep Space Blue
    vec3 color2 = vec3(0.0, 1.0, 1.0);   // Neon Cyan
    vec3 color3 = vec3(1.0, 0.0, 1.0);   // Blinding Magenta
    vec3 color4 = vec3(1.0, 1.0, 1.0);   // Critical White
    
    if (t < 0.33) {
        return mix(color1, color2, t / 0.33);
    } else if (t < 0.66) {
        return mix(color2, color3, (t - 0.33) / 0.33);
    } else {
        return mix(color3, color4, (t - 0.66) / 0.34);
    }
}

void main() {
    float temp = texture(u_texture, v_uv).r;
    float normalized = (temp - u_minTemp) / (u_maxTemp - u_minTemp);
    vec3 color = thermalPalette(normalized);
    
    // Add scanline effect
    float scanline = sin(v_uv.y * 800.0) * 0.04;
    color += scanline;
    
    fragColor = vec4(color, 1.0);
}
`;

export const PHOTONIC_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in float a_intensity;
in vec3 a_color;
uniform mat3 u_projection;
out float v_intensity;
out vec3 v_color;
void main() {
    vec3 proj = u_projection * a_position;
    gl_Position = vec4(proj.xy, 0.0, 1.0);
    gl_PointSize = proj.z * 2.0;
    v_intensity = a_intensity;
    v_color = a_color;
}
`;

export const PHOTONIC_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in float v_intensity;
in vec3 v_color;
out vec4 fragColor;

void main() {
    // Circular glow for waveguide nodes
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    
    float alpha = (1.0 - dist * 2.0) * v_intensity;
    // Add interference pattern ripple
    float ripple = sin(dist * 20.0 - performance.now() * 0.005) * 0.2 + 0.8;
    alpha *= ripple;
    
    fragColor = vec4(v_color, alpha);
}
`;
