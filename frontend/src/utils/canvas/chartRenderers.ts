/**
 * NEXUS-OMEGA FRONTEND STAGE 2
 * Module: Canvas Chart Renderers
 * Purpose: High-performance canvas rendering utilities for PnL waterfall and risk gauges.
 * Performance: Uses double-buffering technique, requestAnimationFrame, minimal allocations.
 */

export interface Point {
  x: number;
  y: number;
}

export interface RenderConfig {
  width: number;
  height: number;
  padding: { top: number; right: number; bottom: number; left: number };
}

/**
 * Renders a glowing PnL waterfall chart with gradient fill.
 * Uses path-based rendering for smooth curves and performance.
 */
export function renderPnLWaterfall(
  ctx: CanvasRenderingContext2D,
  data: number[],
  config: RenderConfig,
  isProfit: boolean
): void {
  const { width, height, padding } = config;
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  if (data.length < 2) return;

  // Calculate min/max for scaling
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < minVal) minVal = data[i];
    if (data[i] > maxVal) maxVal = data[i];
  }

  const range = maxVal - minVal || 1;
  const zeroY = padding.top + ((maxVal - 0) / range) * chartHeight;

  // Create gradient
  const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  if (isProfit) {
    gradient.addColorStop(0, 'rgba(0, 255, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 255, 255, 0.05)');
  } else {
    gradient.addColorStop(0, 'rgba(255, 0, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(255, 0, 255, 0.05)');
  }

  // Draw filled area
  ctx.beginPath();
  ctx.moveTo(padding.left, zeroY);

  const stepX = chartWidth / (data.length - 1);
  for (let i = 0; i < data.length; i++) {
    const x = padding.left + i * stepX;
    const normalizedY = (data[i] - minVal) / range;
    const y = padding.top + (1 - normalizedY) * chartHeight;
    
    if (i === 0) {
      ctx.lineTo(x, y);
    } else {
      // Smooth curve using quadratic bezier
      const prevX = padding.left + (i - 1) * stepX;
      const prevY = padding.top + (1 - (data[i - 1] - minVal) / range) * chartHeight;
      const cpX = (prevX + x) / 2;
      ctx.quadraticCurveTo(cpX, prevY, x, y);
    }
  }

  ctx.lineTo(padding.left + chartWidth, zeroY);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Draw glow line
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = padding.left + i * stepX;
    const normalizedY = (data[i] - minVal) / range;
    const y = padding.top + (1 - normalizedY) * chartHeight;
    
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      const prevX = padding.left + (i - 1) * stepX;
      const prevY = padding.top + (1 - (data[i - 1] - minVal) / range) * chartHeight;
      const cpX = (prevX + x) / 2;
      ctx.quadraticCurveTo(cpX, prevY, x, y);
    }
  }

  const lineColor = isProfit ? '#00ffff' : '#ff00ff';
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.shadowColor = lineColor;
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Draw zero line
  if (zeroY >= padding.top && zeroY <= height - padding.bottom) {
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(width - padding.right, zeroY);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

/**
 * Renders a circular CVaR speedometer gauge.
 * Supports critical zone highlighting and animated needle.
 */
export function renderCVaRSpeedometer(
  ctx: CanvasRenderingContext2D,
  currentValue: number,
  maxValue: number,
  criticalThreshold: number,
  config: RenderConfig
): void {
  const { width, height } = config;
  const centerX = width / 2;
  const centerY = height / 2 + 20;
  const radius = Math.min(width, height) / 2 - 40;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // Draw background arc
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, Math.PI * 0.75, Math.PI * 2.25);
  ctx.strokeStyle = 'rgba(50, 50, 60, 0.5)';
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Calculate angle for current value
  const totalAngle = Math.PI * 1.5;
  const startAngle = Math.PI * 0.75;
  const valueRatio = Math.min(currentValue / maxValue, 1);
  const currentAngle = startAngle + valueRatio * totalAngle;

  // Determine if in critical zone
  const criticalRatio = criticalThreshold / maxValue;
  const criticalAngle = startAngle + criticalRatio * totalAngle;
  const isCritical = currentValue >= criticalThreshold;

  // Draw safe zone arc (green to yellow)
  if (valueRatio > 0) {
    const safeEndAngle = Math.min(currentAngle, criticalAngle);
    if (safeEndAngle > startAngle) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, startAngle, safeEndAngle);
      ctx.strokeStyle = isCritical ? 'rgba(255, 200, 0, 0.8)' : 'rgba(0, 255, 255, 0.8)';
      ctx.lineWidth = 20;
      ctx.lineCap = 'round';
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 15;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  // Draw critical zone arc (red)
  if (valueRatio > criticalRatio) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, criticalAngle, currentAngle);
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(255, 0, 0, 0.8)';
    ctx.shadowBlur = 20;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Draw tick marks
  const tickCount = 10;
  for (let i = 0; i <= tickCount; i++) {
    const tickAngle = startAngle + (i / tickCount) * totalAngle;
    const tickInner = radius - 25;
    const tickOuter = radius - 15;
    
    const x1 = centerX + Math.cos(tickAngle) * tickInner;
    const y1 = centerY + Math.sin(tickAngle) * tickInner;
    const x2 = centerX + Math.cos(tickAngle) * tickOuter;
    const y2 = centerY + Math.sin(tickAngle) * tickOuter;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = i >= criticalRatio * tickCount ? 'rgba(255, 100, 100, 0.5)' : 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw needle
  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(currentAngle - Math.PI / 2);

  ctx.beginPath();
  ctx.moveTo(0, -5);
  ctx.lineTo(radius - 10, 0);
  ctx.lineTo(0, 5);
  ctx.closePath();
  ctx.fillStyle = isCritical ? '#ff0000' : '#00ffff';
  ctx.shadowColor = ctx.fillStyle;
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.shadowBlur = 0;

  // Center cap
  ctx.beginPath();
  ctx.arc(0, 0, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  ctx.restore();

  // Draw value text
  ctx.fillStyle = isCritical ? '#ff4444' : '#00ffff';
  ctx.font = 'bold 24px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(currentValue.toFixed(4), centerX, centerY + radius * 0.5);

  // Draw label
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.font = '12px "JetBrains Mono", monospace';
  ctx.fillText('CVaR (99%)', centerX, centerY + radius * 0.7);

  // Critical warning pulse
  if (isCritical) {
    const pulseAlpha = (Math.sin(Date.now() * 0.005) + 1) / 2 * 0.3;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius + 10, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255, 0, 0, ${pulseAlpha})`;
    ctx.lineWidth = 3;
    ctx.stroke();
  }
}

/**
 * Double-buffer helper for canvas rendering.
 * Returns an offscreen canvas for rendering before blitting to main canvas.
 */
export function createOffscreenCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}
