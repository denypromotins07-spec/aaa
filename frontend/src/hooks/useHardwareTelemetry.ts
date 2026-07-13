'use client';

import { useEffect, useRef, useCallback } from 'react';

// Binary schema matches Rust backend exactly:
// Header: [magic: u32, version: u8, msgType: u8, seqId: u32, timestamp: u64]
// Payload: depends on message type

const MAGIC_HEADER = 0xdeadc0de;
const THERMAL_MSG_TYPE = 0x01;
const PHOTONIC_MSG_TYPE = 0x02;

interface HardwareTelemetry {
  thermalGrid: Float32Array;
  photonicPhases: Float32Array;
  fpgaHealth: number;
  timestamp: number;
}

export function useHardwareTelemetry() {
  const wsRef = useRef<WebSocket | null>(null);
  const dataRef = useRef<HardwareTelemetry>({
    thermalGrid: new Float32Array(64 * 64),
    photonicPhases: new Float32Array(64),
    fpgaHealth: 1.0,
    timestamp: 0,
  });
  const seqIdRef = useRef<number>(0);
  const listenersRef = useRef<Set<(data: HardwareTelemetry) => void>>(new Set());

  const validateEnvelope = useCallback((dataView: DataView): boolean => {
    if (dataView.byteLength < 20) {
      console.error('Packet too small for header');
      return false;
    }

    const magic = dataView.getUint32(0, true); // Little-endian
    if (magic !== MAGIC_HEADER) {
      console.error(`Invalid magic header: expected ${MAGIC_HEADER.toString(16)}, got ${magic.toString(16)}`);
      return false;
    }

    const version = dataView.getUint8(4);
    if (version !== 1) {
      console.error(`Unsupported protocol version: ${version}`);
      return false;
    }

    return true;
  }, []);

  const parseThermalData = useCallback((dataView: DataView, offset: number): Float32Array => {
    const gridSize = 64;
    const totalElements = gridSize * gridSize;
    const thermalData = new Float32Array(totalElements);

    // Rust sends u16 fixed-point (value * 100), convert to float
    for (let i = 0; i < totalElements; i++) {
      const rawValue = dataView.getUint16(offset + i * 2, true);
      thermalData[i] = rawValue / 100.0; // Convert fixed-point to float
    }

    return thermalData;
  }, []);

  const parsePhotonicData = useCallback((dataView: DataView, offset: number): Float32Array => {
    const numPhases = 64;
    const phases = new Float32Array(numPhases);

    // Rust sends f32 directly
    for (let i = 0; i < numPhases; i++) {
      phases[i] = dataView.getFloat32(offset + i * 4, true);
    }

    return phases;
  }, []);

  useEffect(() => {
    const connect = () => {
      wsRef.current = new WebSocket('ws://localhost:8080/ws/hardware');

      wsRef.current.binaryType = 'arraybuffer';

      wsRef.current.onopen = () => {
        console.log('[HARDWARE_WS] Connected');
      };

      wsRef.current.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer)) {
          console.warn('[HARDWARE_WS] Received non-binary data');
          return;
        }

        const dataView = new DataView(event.data);

        if (!validateEnvelope(dataView)) {
          return;
        }

        const msgType = dataView.getUint8(5);
        const seqId = dataView.getUint32(6, true);
        const timestamp = Number(dataView.getBigUint64(10, true));

        // Check sequence ID for gaps (packet loss detection)
        if (seqIdRef.current > 0 && seqId !== seqIdRef.current + 1) {
          console.warn(`[HARDWARE_WS] Packet gap detected: expected ${seqIdRef.current + 1}, got ${seqId}`);
        }
        seqIdRef.current = seqId;

        // Parse payload based on message type
        const payloadOffset = 18; // Header size

        if (msgType === THERMAL_MSG_TYPE) {
          dataRef.current.thermalGrid = parseThermalData(dataView, payloadOffset);
        } else if (msgType === PHOTONIC_MSG_TYPE) {
          dataRef.current.photonicPhases = parsePhotonicData(dataView, payloadOffset);
        }

        dataRef.current.timestamp = timestamp;

        // Notify all listeners
        listenersRef.current.forEach((listener) => {
          try {
            listener({ ...dataRef.current });
          } catch (error) {
            console.error('[HARDWARE_WS] Listener error:', error);
          }
        });
      };

      wsRef.current.onerror = (error) => {
        console.error('[HARDWARE_WS] Error:', error);
      };

      wsRef.current.onclose = () => {
        console.log('[HARDWARE_WS] Disconnected, reconnecting...');
        setTimeout(connect, 1000);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [validateEnvelope, parseThermalData, parsePhotonicData]);

  const subscribe = useCallback((callback: (data: HardwareTelemetry) => void) => {
    listenersRef.current.add(callback);
    return () => {
      listenersRef.current.delete(callback);
    };
  }, []);

  const getData = useCallback(() => {
    return { ...dataRef.current };
  }, []);

  return {
    subscribe,
    getData,
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}
