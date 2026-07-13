'use client';

import React, { useEffect, useRef } from 'react';

interface RaftEvent {
  term: number;
  eventType: 'election_start' | 'leader_elected' | 'heartbeat' | 'vote_request';
  nodeId: string;
  timestamp: number;
}

interface RaftConsensusTimelineProps {
  width: number;
  height: number;
}

export default function RaftConsensusTimeline({ width, height }: RaftConsensusTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const eventsRef = useRef<RaftEvent[]>([]);
  const animationFrameRef = useRef<number>(0);
  const scrollOffsetRef = useRef<number>(0);

  // Generate mock Raft events
  useEffect(() => {
    const generateEvents = () => {
      const events: RaftEvent[] = [];
      const baseTime = Date.now();
      
      for (let i = 0; i < 50; i++) {
        const eventTypeRand = Math.random();
        let eventType: RaftEvent['eventType'] = 'heartbeat';
        if (eventTypeRand > 0.9) eventType = 'election_start';
        else if (eventTypeRand > 0.8) eventType = 'leader_elected';
        else if (eventTypeRand > 0.7) eventType = 'vote_request';

        events.push({
          term: Math.floor(i / 10) + 1,
          eventType,
          nodeId: `node-${Math.floor(Math.random() * 10)}`,
          timestamp: baseTime - (50 - i) * 1000,
        });
      }
      
      eventsRef.current = events;
    };

    generateEvents();

    // Add new events periodically
    const eventInterval = setInterval(() => {
      const lastTerm = eventsRef.current.length > 0 
        ? eventsRef.current[eventsRef.current.length - 1].term 
        : 1;
      
      const eventTypeRand = Math.random();
      let eventType: RaftEvent['eventType'] = 'heartbeat';
      if (eventTypeRand > 0.9) eventType = 'election_start';
      else if (eventTypeRand > 0.8) eventType = 'leader_elected';
      else if (eventTypeRand > 0.7) eventType = 'vote_request';

      eventsRef.current.push({
        term: lastTerm + (eventType === 'election_start' || eventType === 'leader_elected' ? 1 : 0),
        eventType,
        nodeId: `node-${Math.floor(Math.random() * 10)}`,
        timestamp: Date.now(),
      });

      // Keep only last 100 events
      if (eventsRef.current.length > 100) {
        eventsRef.current.shift();
      }
    }, 2000);

    return () => clearInterval(eventInterval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Background
      ctx.fillStyle = 'rgba(10, 10, 12, 0.5)';
      ctx.fillRect(0, 0, width, height);

      const events = eventsRef.current;
      const itemHeight = 20;
      const totalHeight = events.length * itemHeight;
      
      // Auto-scroll to latest
      const maxScroll = Math.max(0, totalHeight - height);
      if (scrollOffsetRef.current > maxScroll) {
        scrollOffsetRef.current = maxScroll;
      }
      scrollOffsetRef.current += (maxScroll - scrollOffsetRef.current) * 0.1;

      ctx.save();
      ctx.translate(0, -scrollOffsetRef.current);

      events.forEach((event, index) => {
        const y = index * itemHeight;
        
        // Event color based on type
        let color = '#06b6d4'; // Cyan - heartbeat
        if (event.eventType === 'election_start') color = '#f97316'; // Orange
        if (event.eventType === 'leader_elected') color = '#fbbf24'; // Gold
        if (event.eventType === 'vote_request') color = '#a855f7'; // Purple

        // Draw event bar
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.7;
        ctx.fillRect(10, y + 2, width - 20, itemHeight - 4);
        
        // Draw term indicator
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText(`T${event.term}`, 15, y + 14);

        // Draw event type
        ctx.fillStyle = '#000000';
        ctx.fillText(event.eventType.toUpperCase().replace('_', ' '), 60, y + 14);

        // Draw node ID
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(event.nodeId, width - 80, y + 14);

        // Draw timestamp
        const timeStr = new Date(event.timestamp).toLocaleTimeString();
        ctx.fillStyle = '#64748b';
        ctx.fillText(timeStr, width - 140, y + 14);

        ctx.globalAlpha = 1;
      });

      ctx.restore();

      // Draw gradient overlay at top and bottom
      const gradientTop = ctx.createLinearGradient(0, 0, 0, 40);
      gradientTop.addColorStop(0, 'rgba(10, 10, 12, 1)');
      gradientTop.addColorStop(1, 'rgba(10, 10, 12, 0)');
      ctx.fillStyle = gradientTop;
      ctx.fillRect(0, 0, width, 40);

      const gradientBottom = ctx.createLinearGradient(0, height - 40, 0, height);
      gradientBottom.addColorStop(0, 'rgba(10, 10, 12, 0)');
      gradientBottom.addColorStop(1, 'rgba(10, 10, 12, 1)');
      ctx.fillStyle = gradientBottom;
      ctx.fillRect(0, height - 40, width, 40);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height]);

  return (
    <div className="relative w-full h-full glass-panel rounded-lg overflow-hidden border border-cyan-900/30">
      <div className="absolute top-2 left-3 z-10 font-mono text-xs text-cyan-400 tracking-wider pointer-events-none">
        RAFT_CONSENSUS :: EVENT_TIMELINE
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
    </div>
  );
}
