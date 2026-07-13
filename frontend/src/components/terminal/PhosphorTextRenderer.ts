import type { LogEntry } from '../BaseRealityTerminal';

interface ColorScheme {
  primary: string;
  secondary: string;
  glow: string;
}

const levelColors: Record<string, ColorScheme> = {
  INFO: { primary: '#00ff88', secondary: '#006633', glow: 'rgba(0, 255, 136, 0.3)' },
  WARN: { primary: '#ffaa00', secondary: '#664400', glow: 'rgba(255, 170, 0, 0.3)' },
  ERROR: { primary: '#ff4444', secondary: '#661111', glow: 'rgba(255, 68, 68, 0.3)' },
  CRITICAL: { primary: '#ff00ff', secondary: '#660066', glow: 'rgba(255, 0, 255, 0.4)' },
  ONTOLOGICAL: { primary: '#ffffff', secondary: '#4444ff', glow: 'rgba(255, 255, 255, 0.5)' },
};

export class PhosphorTextRenderer {
  private ctx: CanvasRenderingContext2D;
  private fontCache: Map<string, boolean>;

  constructor(ctx: CanvasRenderingContext2D) {
    this.ctx = ctx;
    this.fontCache = new Map();
  }

  renderLine(log: LogEntry, x: number, y: number, maxWidth: number): void {
    const colors = levelColors[log.level] || levelColors.INFO;
    
    // Draw background highlight for critical/ontological
    if (log.level === 'CRITICAL' || log.level === 'ONTOLOGICAL') {
      this.ctx.fillStyle = colors.glow;
      this.ctx.fillRect(x - 5, y - 12, maxWidth + 10, 18);
    }

    // Render timestamp
    this.ctx.font = '11px "JetBrains Mono", monospace';
    this.ctx.fillStyle = colors.secondary;
    const timeStr = new Date(log.timestamp).toISOString().substr(11, 12);
    this.ctx.fillText(timeStr, x, y);

    // Render level badge
    const levelX = x + 130;
    this.ctx.fillStyle = colors.primary;
    this.ctx.font = 'bold 11px "JetBrains Mono", monospace';
    this.ctx.fillText(`[${log.level}]`, levelX, y);

    // Render source
    const sourceX = levelX + 80;
    this.ctx.fillStyle = '#666688';
    this.ctx.font = '10px "JetBrains Mono", monospace';
    this.ctx.fillText(log.source, sourceX, y);

    // Render message with glow effect for high-priority logs
    const msgX = sourceX + 120;
    const maxMsgWidth = maxWidth - (msgX - x);
    
    if (log.level === 'CRITICAL' || log.level === 'ONTOLOGICAL') {
      // Add glow effect
      this.ctx.shadowColor = colors.primary;
      this.ctx.shadowBlur = 8;
    } else {
      this.ctx.shadowBlur = 0;
    }

    this.ctx.fillStyle = colors.primary;
    this.ctx.font = '12px "JetBrains Mono", monospace';
    
    // Truncate message if too long
    let message = log.message;
    if (this.ctx.measureText(message).width > maxMsgWidth) {
      while (this.ctx.measureText(message + '...').width > maxMsgWidth && message.length > 0) {
        message = message.slice(0, -1);
      }
      message = message + '...';
    }
    
    this.ctx.fillText(message, msgX, y);
    
    // Reset shadow
    this.ctx.shadowBlur = 0;

    // Draw phosphor bloom for special levels
    if (log.level === 'ONTOLOGICAL') {
      this.drawPhosphorBloom(msgX, y - 8, message, colors.primary);
    }
  }

  private drawPhosphorBloom(x: number, y: number, text: string, color: string): void {
    const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 100);
    gradient.addColorStop(0, color.replace(')', ', 0.2)').replace('rgb', 'rgba'));
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    
    this.ctx.fillStyle = gradient;
    this.ctx.beginPath();
    this.ctx.arc(x + this.ctx.measureText(text).width / 2, y, 60, 0, Math.PI * 2);
    this.ctx.fill();
  }

  drawScanlines(width: number, height: number): void {
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.03)';
    
    // Horizontal scanlines
    for (let y = 0; y < height; y += 3) {
      this.ctx.fillRect(0, y, width, 1);
    }
  }

  drawVignette(width: number, height: number): void {
    const gradient = this.ctx.createRadialGradient(
      width / 2,
      height / 2,
      Math.min(width, height) * 0.3,
      width / 2,
      height / 2,
      Math.max(width, height) * 0.8
    );
    
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
    gradient.addColorStop(0.5, 'rgba(0, 0, 0, 0.1)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0.4)');
    
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, width, height);
  }

  clear(): void {
    const canvas = this.ctx.canvas;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}
