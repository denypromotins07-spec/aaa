// High-performance particle system for network packet visualization
// Uses object pooling to prevent GC allocations during steady-state operation

export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
  type: 'packet' | 'splice' | 'drop';
  active: boolean;
}

export class ParticlePool {
  private pool: Particle[];
  private activeCount: number = 0;

  constructor(maxParticles: number = 1000) {
    this.pool = [];
    for (let i = 0; i < maxParticles; i++) {
      this.pool.push({
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        life: 0,
        maxLife: 1,
        color: '#06b6d4',
        size: 2,
        type: 'packet',
        active: false,
      });
    }
  }

  spawn(
    x: number,
    y: number,
    vx: number,
    vy: number,
    life: number,
    color: string,
    size: number,
    type: 'packet' | 'splice' | 'drop'
  ): Particle | null {
    // Find inactive particle
    const particle = this.pool.find((p) => !p.active);
    if (!particle) return null;

    particle.x = x;
    particle.y = y;
    particle.vx = vx;
    particle.vy = vy;
    particle.life = 0;
    particle.maxLife = life;
    particle.color = color;
    particle.size = size;
    particle.type = type;
    particle.active = true;

    this.activeCount++;
    return particle;
  }

  update(deltaTime: number): void {
    for (const particle of this.pool) {
      if (!particle.active) continue;

      particle.life += deltaTime;
      particle.x += particle.vx * deltaTime;
      particle.y += particle.vy * deltaTime;

      // Fade out near end of life
      const lifeRatio = particle.life / particle.maxLife;
      if (lifeRatio > 0.8) {
        particle.size *= 0.95;
      }

      if (particle.life >= particle.maxLife) {
        particle.active = false;
        this.activeCount--;
      }
    }
  }

  getActiveParticles(): Particle[] {
    return this.pool.filter((p) => p.active);
  }

  reset(): void {
    for (const particle of this.pool) {
      particle.active = false;
    }
    this.activeCount = 0;
  }

  getActiveCount(): number {
    return this.activeCount;
  }
}

// Pre-defined particle styles for different network events
export const PARTICLE_STYLES = {
  udpIncoming: {
    color: '#06b6d4', // Cyan
    speed: 300,
    size: 3,
    life: 2,
  },
  tcpOutgoing: {
    color: '#22c55e', // Green
    speed: 250,
    size: 2,
    life: 2.5,
  },
  xdpSplice: {
    color: '#e879f9', // Magenta
    speed: 150,
    size: 5,
    life: 1,
  },
  packetDrop: {
    color: '#ef4444', // Red
    speed: 0,
    size: 4,
    life: 0.5,
  },
};

// Render particles to canvas with motion blur effect
export function renderParticles(
  ctx: CanvasRenderingContext2D,
  particles: Particle[],
  trailLength: number = 10
): void {
  for (const particle of particles) {
    ctx.beginPath();

    // Draw trail
    const trailX = particle.x - particle.vx * trailLength * 0.01;
    const trailY = particle.y - particle.vy * trailLength * 0.01;

    const gradient = ctx.createLinearGradient(trailX, trailY, particle.x, particle.y);
    gradient.addColorStop(0, 'transparent');
    gradient.addColorStop(1, particle.color);

    ctx.strokeStyle = gradient;
    ctx.lineWidth = particle.size;
    ctx.lineCap = 'round';
    ctx.moveTo(trailX, trailY);
    ctx.lineTo(particle.x, particle.y);
    ctx.stroke();

    // Draw particle head
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
    ctx.fillStyle = particle.color;
    ctx.fill();

    // Add glow effect
    ctx.shadowColor = particle.color;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

// Spawn XDP packet flow particles
export function spawnXDPPacket(
  pool: ParticlePool,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  isSplice: boolean = false
): void {
  const dx = endX - startX;
  const dy = endY - startY;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const style = isSplice ? PARTICLE_STYLES.xdpSplice : PARTICLE_STYLES.udpIncoming;

  const vx = (dx / distance) * style.speed;
  const vy = (dy / distance) * style.speed;

  pool.spawn(startX, startY, vx, vy, style.life, style.color, style.size, isSplice ? 'splice' : 'packet');
}
