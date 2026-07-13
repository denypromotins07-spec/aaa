export interface SpikePoint {
  electrodeId: number;
  timestamp: number;
  amplitude: number;
  type: 'excitatory' | 'inhibitory';
}

export class SpikeRingBuffer {
  private buffer: Float32Array;
  private head: number = 0;
  private tail: number = 0;
  private count: number = 0;
  private readonly capacity: number;

  constructor(capacity: number = 50000) {
    this.capacity = capacity;
    // Each spike: [electrodeId, timestamp, amplitude, type(0/1)]
    this.buffer = new Float32Array(capacity * 4);
  }

  push(spike: SpikePoint): void {
    const idx = this.head * 4;
    this.buffer[idx] = spike.electrodeId;
    this.buffer[idx + 1] = spike.timestamp;
    this.buffer[idx + 2] = spike.amplitude;
    this.buffer[idx + 3] = spike.type === 'excitatory' ? 0 : 1;

    this.head = (this.head + 1) % this.capacity;
    
    if (this.count === this.capacity) {
      this.tail = (this.tail + 1) % this.capacity;
    } else {
      this.count++;
    }
  }

  pushBatch(spikes: SpikePoint[]): void {
    for (const spike of spikes) {
      this.push(spike);
    }
  }

  getRange(startTime: number, endTime: number): SpikePoint[] {
    const result: SpikePoint[] = [];
    let idx = this.tail;
    let visited = 0;

    while (visited < this.count) {
      const base = idx * 4;
      const timestamp = this.buffer[base + 1];
      
      if (timestamp >= startTime && timestamp <= endTime) {
        result.push({
          electrodeId: Math.floor(this.buffer[base]),
          timestamp,
          amplitude: this.buffer[base + 2],
          type: this.buffer[base + 3] === 0 ? 'excitatory' : 'inhibitory',
        });
      }
      
      idx = (idx + 1) % this.capacity;
      visited++;
    }

    return result;
  }

  getAll(): SpikePoint[] {
    const result: SpikePoint[] = [];
    let idx = this.tail;
    let visited = 0;

    while (visited < this.count) {
      const base = idx * 4;
      result.push({
        electrodeId: Math.floor(this.buffer[base]),
        timestamp: this.buffer[base + 1],
        amplitude: this.buffer[base + 2],
        type: this.buffer[base + 3] === 0 ? 'excitatory' : 'inhibitory',
      });
      
      idx = (idx + 1) % this.capacity;
      visited++;
    }

    return result;
  }

  getCount(): number {
    return this.count;
  }

  clear(): void {
    this.head = 0;
    this.tail = 0;
    this.count = 0;
    this.buffer.fill(0);
  }
}

export function createSpikeBuffer(capacity?: number): SpikeRingBuffer {
  return new SpikeRingBuffer(capacity);
}
