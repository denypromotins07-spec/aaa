'use client';

import React, { useEffect, useRef } from 'react';

interface SankeyLink {
  source: string;
  target: string;
  value: number;
  tcaSlippage: number; // negative = improvement, positive = slippage
}

interface SankeyNode {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DarkPoolSankeyProps {
  width: number;
  height: number;
}

export default function DarkPoolSankey({ width, height }: DarkPoolSankeyProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const linksRef = useRef<SankeyLink[]>([]);
  const nodesRef = useRef<SankeyNode[]>([]);
  const animationFrameRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  // Initialize Sankey diagram structure
  useEffect(() => {
    // Define nodes in columns: Meta-Order -> Venues -> Execution
    nodesRef.current = [
      // Source (Meta-Order)
      { id: 'meta', label: 'META_ORDER', x: 50, y: height / 2 - 40, width: 80, height: 80 },
      
      // Venues (middle column)
      { id: 'lit', label: 'LIT_EXCHANGE', x: width * 0.35, y: 80, width: 100, height: 60 },
      { id: 'dark', label: 'DARK_POOL', x: width * 0.35, y: height / 2 - 30, width: 100, height: 60 },
      { id: 'rfq', label: 'RFQ_VENUE', x: width * 0.35, y: height - 140, width: 100, height: 60 },
      
      // Execution outcomes (right column)
      { id: 'filled', label: 'FILLED', x: width - 150, y: 100, width: 80, height: 60 },
      { id: 'partial', label: 'PARTIAL', x: width - 150, y: height / 2 - 30, width: 80, height: 60 },
      { id: 'cancelled', label: 'CANCELLED', x: width - 150, y: height - 140, width: 80, height: 60 },
    ];

    // Initial link values
    linksRef.current = [
      { source: 'meta', target: 'lit', value: 40, tcaSlippage: -0.002 },
      { source: 'meta', target: 'dark', value: 45, tcaSlippage: 0.001 },
      { source: 'meta', target: 'rfq', value: 15, tcaSlippage: -0.005 },
      
      { source: 'lit', target: 'filled', value: 35, tcaSlippage: -0.001 },
      { source: 'lit', target: 'partial', value: 5, tcaSlippage: 0.003 },
      
      { source: 'dark', target: 'filled', value: 30, tcaSlippage: 0.002 },
      { source: 'dark', target: 'partial', value: 10, tcaSlippage: 0.005 },
      { source: 'dark', target: 'cancelled', value: 5, tcaSlippage: 0 },
      
      { source: 'rfq', target: 'filled', value: 12, tcaSlippage: -0.004 },
      { source: 'rfq', target: 'partial', value: 3, tcaSlippage: -0.001 },
    ];

    // Simulate dynamic flow updates
    const updateInterval = setInterval(() => {
      linksRef.current = linksRef.current.map((link) => ({
        ...link,
        value: Math.max(5, link.value + (Math.random() - 0.5) * 10),
        tcaSlippage: link.tcaSlippage + (Math.random() - 0.5) * 0.001,
      }));
    }, 1000);

    return () => clearInterval(updateInterval);
  }, [width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      timeRef.current += 0.016;

      ctx.clearRect(0, 0, width, height);
      
      // Background
      ctx.fillStyle = 'rgba(10, 10, 12, 0.3)';
      ctx.fillRect(0, 0, width, height);

      const nodes = nodesRef.current;
      const links = linksRef.current;

      // Calculate total value for normalization
      const totalValue = links
        .filter((l) => l.source === 'meta')
        .reduce((sum, l) => sum + l.value, 0);

      // Draw links first (so they appear behind nodes)
      links.forEach((link) => {
        const sourceNode = nodes.find((n) => n.id === link.source);
        const targetNode = nodes.find((n) => n.id === link.target);
        
        if (!sourceNode || !targetNode) return;

        const linkWidth = (link.value / totalValue) * Math.min(sourceNode.height, targetNode.height);
        
        // Color based on TCA slippage
        const slippageColor = getSlippageColor(link.tcaSlippage);
        
        // Create curved path
        const startX = sourceNode.x + sourceNode.width;
        const startY = sourceNode.y + sourceNode.height / 2;
        const endX = targetNode.x;
        const endY = targetNode.y + targetNode.height / 2;
        
        // Control points for bezier curve
        const cp1x = startX + (endX - startX) * 0.5;
        const cp1y = startY;
        const cp2x = endX - (endX - startX) * 0.5;
        const cp2y = endY;

        // Draw flow with gradient
        const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
        gradient.addColorStop(0, slippageColor.replace('0.7', '0.3'));
        gradient.addColorStop(0.5, slippageColor.replace('0.7', '0.6'));
        gradient.addColorStop(1, slippageColor.replace('0.7', '0.3'));

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);
        
        // Draw thick line for flow
        ctx.strokeStyle = gradient;
        ctx.lineWidth = linkWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Add animated particles along the path
        const particlePos = (timeRef.current * 0.5) % 1;
        const px = bezierPoint(startX, cp1x, cp2x, endX, particlePos);
        const py = bezierPoint(startY, cp1y, cp2y, endY, particlePos);
        
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      });

      // Draw nodes
      nodes.forEach((node) => {
        // Node background
        ctx.fillStyle = 'rgba(30, 30, 40, 0.9)';
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1;
        ctx.fillRect(node.x, node.y, node.width, node.height);
        ctx.strokeRect(node.x, node.y, node.width, node.height);

        // Node label
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(node.label, node.x + node.width / 2, node.y + node.height / 2 + 3);
      });

      // Draw legend
      drawLegend(ctx);

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
        DARK_POOL_ROUTING :: TCA_ANALYSIS
      </div>
      <div className="absolute bottom-2 right-3 z-10 font-mono text-xs text-green-400 pointer-events-none">
        WIDTH=VOLUME | COLOR=TCA_SHORTFALL
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

function getSlippageColor(slippage: number): string {
  // Green for price improvement (negative slippage), red for slippage (positive)
  if (slippage < -0.003) return 'rgba(34, 197, 94, 0.7)'; // Deep green
  if (slippage < 0) return 'rgba(134, 239, 172, 0.7)'; // Light green
  if (slippage < 0.003) return 'rgba(250, 204, 21, 0.7)'; // Yellow
  if (slippage < 0.01) return 'rgba(249, 115, 22, 0.7)'; // Orange
  return 'rgba(239, 68, 68, 0.7)'; // Red
}

function bezierPoint(p0: number, p1: number, p2: number, p3: number, t: number): number {
  // Cubic bezier interpolation
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}

function drawLegend(ctx: CanvasRenderingContext2D) {
  const legendX = 10;
  const legendY = 10;
  const boxSize = 12;
  const spacing = 18;

  ctx.font = '8px JetBrains Mono, monospace';
  ctx.textAlign = 'left';

  const items = [
    { color: 'rgba(34, 197, 94, 0.7)', label: 'PRICE_IMPROVE' },
    { color: 'rgba(250, 204, 21, 0.7)', label: 'NEUTRAL' },
    { color: 'rgba(239, 68, 68, 0.7)', label: 'SLIPPAGE' },
  ];

  items.forEach((item, i) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, legendY + i * spacing, boxSize, boxSize);
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(item.label, legendX + boxSize + 5, legendY + i * spacing + 10);
  });
}
