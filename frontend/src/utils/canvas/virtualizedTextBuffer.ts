/**
 * Virtualized Ring Buffer for high-performance text rendering.
 * Prevents DOM thrashing by maintaining a fixed-size circular buffer
 * and only rendering visible viewport lines.
 */

export interface TextLine {
  id: number;
  content: string;
  timestamp: number;
  priority: 'low' | 'normal' | 'high' | 'critical';
}

export class VirtualizedTextBuffer {
  private buffer: TextLine[];
  private maxSize: number;
  private writeIndex: number;
  private readOffset: number;
  private idCounter: number;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
    this.buffer = new Array<TextLine>(maxSize);
    this.writeIndex = 0;
    this.readOffset = 0;
    this.idCounter = 0;
  }

  /**
   * Push a new line to the buffer (O(1) operation)
   */
  push(content: string, priority: TextLine['priority'] = 'normal'): TextLine {
    const line: TextLine = {
      id: this.idCounter++,
      content,
      timestamp: Date.now(),
      priority,
    };

    this.buffer[this.writeIndex] = line;
    this.writeIndex = (this.writeIndex + 1) % this.maxSize;

    // Auto-adjust read offset if we've wrapped around
    if (this.writeIndex === this.readOffset) {
      this.readOffset = (this.readOffset + 1) % this.maxSize;
    }

    return line;
  }

  /**
   * Get visible lines for rendering (virtualized viewport)
   */
  getVisibleLines(visibleCount: number, scrollOffset: number = 0): TextLine[] {
    const result: TextLine[] = [];
    const totalLines = this.size();
    
    // Calculate start index based on scroll offset
    const startIndex = Math.max(0, totalLines - visibleCount - scrollOffset);
    const endIndex = Math.min(totalLines, startIndex + visibleCount);

    for (let i = startIndex; i < endIndex; i++) {
      const bufferIndex = (this.readOffset + i) % this.maxSize;
      if (this.buffer[bufferIndex]) {
        result.push(this.buffer[bufferIndex]);
      }
    }

    return result;
  }

  /**
   * Get total number of lines in buffer
   */
  size(): number {
    if (this.writeIndex >= this.readOffset) {
      return this.writeIndex - this.readOffset;
    }
    return this.maxSize - this.readOffset + this.writeIndex;
  }

  /**
   * Clear all lines
   */
  clear(): void {
    this.buffer = new Array<TextLine>(this.maxSize);
    this.writeIndex = 0;
    this.readOffset = 0;
  }

  /**
   * Scroll up/down
   */
  scroll(delta: number): void {
    this.readOffset = Math.max(
      0,
      Math.min(this.readOffset + delta, this.maxSize - this.size())
    );
  }

  /**
   * Get line by ID (for searching/highlighting)
   */
  getById(id: number): TextLine | undefined {
    for (let i = 0; i < this.maxSize; i++) {
      if (this.buffer[i]?.id === id) {
        return this.buffer[i];
      }
    }
    return undefined;
  }

  /**
   * Find lines matching a pattern
   */
  find(pattern: RegExp, limit: number = 50): TextLine[] {
    const results: TextLine[] = [];
    for (let i = 0; i < this.maxSize && results.length < limit; i++) {
      const line = this.buffer[i];
      if (line && pattern.test(line.content)) {
        results.push(line);
      }
    }
    return results;
  }

  /**
   * Export recent lines for debugging/logging
   */
  export(count: number = 100): string {
    const lines = this.getVisibleLines(count);
    return lines.map(l => `[${new Date(l.timestamp).toISOString()}] ${l.content}`).join('\n');
  }
}

/**
 * Double-buffered text renderer for canvas-based terminals.
 * Uses off-screen canvas to prevent main thread blocking.
 */
export class DoubleBufferedTextRenderer {
  private mainCanvas: HTMLCanvasElement;
  private offscreenCanvas: HTMLCanvasElement;
  private mainCtx: CanvasRenderingContext2D;
  private offscreenCtx: CanvasRenderingContext2D;
  private buffer: VirtualizedTextBuffer;
  private lineHeight: number;
  private needsRedraw: boolean;

  constructor(
    canvas: HTMLCanvasElement,
    bufferSize: number = 1000,
    lineHeight: number = 18
  ) {
    this.mainCanvas = canvas;
    this.lineHeight = lineHeight;
    this.needsRedraw = true;

    // Create offscreen canvas for double buffering
    this.offscreenCanvas = document.createElement('canvas');
    
    const mainCtx = canvas.getContext('2d');
    const offscreenCtx = this.offscreenCanvas.getContext('2d');
    
    if (!mainCtx || !offscreenCtx) {
      throw new Error('Failed to create canvas contexts');
    }
    
    this.mainCtx = mainCtx;
    this.offscreenCtx = offscreenCtx;
    this.buffer = new VirtualizedTextBuffer(bufferSize);
  }

  /**
   * Add text line to buffer
   */
  append(text: string, priority: TextLine['priority'] = 'normal'): void {
    this.buffer.push(text, priority);
    this.needsRedraw = true;
  }

  /**
   * Render visible portion to offscreen canvas, then blit to main
   */
  render(): void {
    if (!this.needsRedraw) return;

    const width = this.mainCanvas.width;
    const height = this.mainCanvas.height;
    const visibleLines = Math.floor(height / this.lineHeight);

    // Resize offscreen canvas if needed
    if (this.offscreenCanvas.width !== width || this.offscreenCanvas.height !== height) {
      this.offscreenCanvas.width = width;
      this.offscreenCanvas.height = height;
    }

    // Clear offscreen canvas
    this.offscreenCtx.fillStyle = '#0a0a0c';
    this.offscreenCtx.fillRect(0, 0, width, height);

    // Get visible lines
    const lines = this.buffer.getVisibleLines(visibleLines);

    // Render to offscreen
    lines.forEach((line, idx) => {
      const y = (idx + 1) * this.lineHeight;
      this.renderLineToContext(line, 10, y, width - 20);
    });

    // Blit to main canvas (single operation)
    this.mainCtx.clearRect(0, 0, width, height);
    this.mainCtx.drawImage(this.offscreenCanvas, 0, 0);

    this.needsRedraw = false;
  }

  private renderLineToContext(
    line: TextLine,
    x: number,
    y: number,
    maxWidth: number
  ): void {
    const ctx = this.offscreenCtx;
    
    // Color based on priority
    const colors = {
      low: '#666688',
      normal: '#00ff88',
      high: '#ffaa00',
      critical: '#ff4444',
    };

    ctx.fillStyle = colors[line.priority];
    ctx.font = '12px "JetBrains Mono", monospace';
    
    // Truncate if too long
    let content = line.content;
    if (ctx.measureText(content).width > maxWidth) {
      while (ctx.measureText(content + '...').width > maxWidth && content.length > 0) {
        content = content.slice(0, -1);
      }
      content += '...';
    }

    ctx.fillText(content, x, y);
  }

  /**
   * Force redraw on next frame
   */
  invalidate(): void {
    this.needsRedraw = true;
  }

  /**
   * Clear buffer and screen
   */
  clear(): void {
    this.buffer.clear();
    this.needsRedraw = true;
  }
}
