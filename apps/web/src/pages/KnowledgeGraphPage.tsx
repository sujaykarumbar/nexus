import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Network, Sparkles, Search, RefreshCw, ZoomIn, ZoomOut, Info,
  ChevronRight, Cpu, Database, TrendingUp, AlertTriangle, FileText,
  GitFork, Target, BarChart2, Circle, Loader2, CheckCircle, XCircle, X
} from 'lucide-react';
import { api } from '../services/api';
import {
  KnowledgeGraphData, GraphNodeData, GraphEdgeData, GraphRAGResult, GraphBuildResult
} from '../types';
import { useAuth } from '../context/AuthContext';

// ─────────────────────────────────────────────────────────────────────────────
// Color & icon map per node type
// ─────────────────────────────────────────────────────────────────────────────
const NODE_TYPE_CONFIG: Record<string, { color: string; bg: string; border: string; icon: React.ElementType }> = {
  dataset:  { color: '#3B82F6', bg: '#1E3A5F', border: '#3B82F6', icon: Database },
  column:   { color: '#10B981', bg: '#064E3B', border: '#10B981', icon: BarChart2 },
  problem:  { color: '#8B5CF6', bg: '#2E1065', border: '#8B5CF6', icon: Target },
  model:    { color: '#F59E0B', bg: '#451A03', border: '#F59E0B', icon: Cpu },
  metric:   { color: '#06B6D4', bg: '#083344', border: '#06B6D4', icon: TrendingUp },
  anomaly:  { color: '#EF4444', bg: '#450A0A', border: '#EF4444', icon: AlertTriangle },
  forecast: { color: '#F97316', bg: '#431407', border: '#F97316', icon: TrendingUp },
  document: { color: '#A855F7', bg: '#2E1065', border: '#A855F7', icon: FileText },
  chunk:    { color: '#6366F1', bg: '#1E1B4B', border: '#6366F1', icon: FileText },
  concept:  { color: '#EC4899', bg: '#500724', border: '#EC4899', icon: Sparkles },
};

const RELATION_COLORS: Record<string, string> = {
  HAS_COLUMN:       '#3B82F6',
  CORRELATES_WITH:  '#10B981',
  TARGET_OF:        '#8B5CF6',
  TRAINED_ON:       '#F59E0B',
  ACHIEVED_METRIC:  '#06B6D4',
  HAS_ANOMALY:      '#EF4444',
  HAS_FORECAST:     '#F97316',
  MENTIONS:         '#A855F7',
  DERIVED_FROM:     '#94A3B8',
  PART_OF:          '#64748B',
  BELONGS_TO:       '#64748B',
};

function getNodeConfig(nodeType: string) {
  return NODE_TYPE_CONFIG[nodeType] || { color: '#94A3B8', bg: '#1E293B', border: '#94A3B8', icon: Circle };
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple force-directed SVG renderer using D3-lite manual layout
// ─────────────────────────────────────────────────────────────────────────────
interface SimNode extends GraphNodeData {
  x: number; y: number; vx: number; vy: number;
}

function useForceLayout(nodes: GraphNodeData[], edges: GraphEdgeData[], width: number, height: number) {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const animRef = useRef<number | null>(null);

  useEffect(() => {
    if (!nodes.length) { setPositions({}); return; }

    const simNodes: SimNode[] = nodes.map((n, i) => ({
      ...n,
      x: width / 2 + Math.cos((i / nodes.length) * 2 * Math.PI) * Math.min(width, height) * 0.35,
      y: height / 2 + Math.sin((i / nodes.length) * 2 * Math.PI) * Math.min(width, height) * 0.35,
      vx: 0, vy: 0,
    }));

    const nodeMap: Record<string, SimNode> = {};
    simNodes.forEach(n => { nodeMap[n.id] = n; });

    let iter = 0;
    const MAX_ITER = 200;
    const REPEL = 8000;
    const ATTRACT = 0.05;
    const CENTER = 0.012;
    const DAMPEN = 0.85;

    const tick = () => {
      if (iter++ > MAX_ITER) {
        const pos: Record<string, { x: number; y: number }> = {};
        simNodes.forEach(n => { pos[n.id] = { x: n.x, y: n.y }; });
        setPositions(pos);
        return;
      }

      // Repulsion
      for (let i = 0; i < simNodes.length; i++) {
        for (let j = i + 1; j < simNodes.length; j++) {
          const a = simNodes[i], b = simNodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist2 = Math.max(dx * dx + dy * dy, 1);
          const force = REPEL / dist2;
          const nx = dx / Math.sqrt(dist2), ny = dy / Math.sqrt(dist2);
          a.vx += force * nx; a.vy += force * ny;
          b.vx -= force * nx; b.vy -= force * ny;
        }
      }

      // Attraction along edges
      edges.forEach(e => {
        const a = nodeMap[e.source_id], b = nodeMap[e.target_id];
        if (!a || !b) return;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = ATTRACT * (dist - 120);
        const nx = dx / dist, ny = dy / dist;
        a.vx += force * nx; a.vy += force * ny;
        b.vx -= force * nx; b.vy -= force * ny;
      });

      // Center gravity
      simNodes.forEach(n => {
        n.vx += (width / 2 - n.x) * CENTER;
        n.vy += (height / 2 - n.y) * CENTER;
        n.vx *= DAMPEN; n.vy *= DAMPEN;
        n.x += n.vx; n.y += n.vy;
        n.x = Math.max(50, Math.min(width - 50, n.x));
        n.y = Math.max(50, Math.min(height - 50, n.y));
      });

      const pos: Record<string, { x: number; y: number }> = {};
      simNodes.forEach(n => { pos[n.id] = { x: n.x, y: n.y }; });
      setPositions(pos);

      animRef.current = requestAnimationFrame(tick);
    };

    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [nodes, edges, width, height]);

  return positions;
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Canvas Component
// ─────────────────────────────────────────────────────────────────────────────
interface GraphCanvasProps {
  nodes: GraphNodeData[];
  edges: GraphEdgeData[];
  selectedNode: GraphNodeData | null;
  highlightIds: Set<string>;
  onNodeClick: (n: GraphNodeData) => void;
  scale: number;
}

const GraphCanvas: React.FC<GraphCanvasProps> = ({ nodes, edges, selectedNode, highlightIds, onNodeClick, scale }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dims, setDims] = useState({ w: 900, h: 580 });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState<{ last: { x: number; y: number } } | null>(null);

  useEffect(() => {
    const el = svgRef.current?.parentElement;
    if (!el) return;
    const ro = new ResizeObserver(e => {
      const { width, height } = e[0].contentRect;
      setDims({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const positions = useForceLayout(nodes, edges, dims.w, dims.h);

  const onMouseDown = (e: React.MouseEvent) => {
    setDragging({ last: { x: e.clientX, y: e.clientY } });
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragging) return;
    setPan(p => ({ x: p.x + e.clientX - dragging.last.x, y: p.y + e.clientY - dragging.last.y }));
    setDragging({ last: { x: e.clientX, y: e.clientY } });
  };
  const onMouseUp = () => setDragging(null);

  const NODE_R = 22;

  return (
    <svg
      ref={svgRef}
      width="100%" height="100%"
      className="cursor-grab active:cursor-grabbing select-none"
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#4B5563" />
        </marker>
        {Object.entries(RELATION_COLORS).map(([rel, color]) => (
          <marker key={rel} id={`arrow-${rel}`} markerWidth="8" markerHeight="8" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill={color} />
          </marker>
        ))}
      </defs>

      <g transform={`translate(${pan.x},${pan.y}) scale(${scale})`}>
        {/* Edges */}
        {edges.map(e => {
          const src = positions[e.source_id];
          const tgt = positions[e.target_id];
          if (!src || !tgt) return null;
          const color = RELATION_COLORS[e.relation] || '#4B5563';
          const isHighlighted = highlightIds.has(e.source_id) || highlightIds.has(e.target_id);
          const opacity = highlightIds.size > 0 ? (isHighlighted ? 1 : 0.15) : 0.55;
          return (
            <g key={e.id} opacity={opacity}>
              <line
                x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
                stroke={color} strokeWidth={isHighlighted ? 2.5 : 1.2}
                strokeDasharray={e.relation === 'CORRELATES_WITH' ? '4,3' : undefined}
                markerEnd={`url(#arrow-${e.relation})`}
              />
              <text
                x={(src.x + tgt.x) / 2}
                y={(src.y + tgt.y) / 2 - 4}
                fill={color} fontSize="7" textAnchor="middle"
                className="pointer-events-none"
                opacity={0.8}
              >
                {e.relation.replace(/_/g, ' ')}
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map(n => {
          const pos = positions[n.id];
          if (!pos) return null;
          const cfg = getNodeConfig(n.node_type);
          const isSelected = selectedNode?.id === n.id;
          const isHighlighted = highlightIds.has(n.id);
          const dimmed = highlightIds.size > 0 && !isHighlighted;

          return (
            <g
              key={n.id}
              transform={`translate(${pos.x},${pos.y})`}
              onClick={(ev) => { ev.stopPropagation(); onNodeClick(n); }}
              className="cursor-pointer"
              opacity={dimmed ? 0.2 : 1}
            >
              {/* Glow ring for selected */}
              {isSelected && (
                <circle r={NODE_R + 6} fill="none" stroke={cfg.color} strokeWidth={2} opacity={0.5}>
                  <animate attributeName="r" values={`${NODE_R + 4};${NODE_R + 8};${NODE_R + 4}`} dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.6;0.2;0.6" dur="2s" repeatCount="indefinite" />
                </circle>
              )}
              <circle
                r={NODE_R}
                fill={cfg.bg}
                stroke={isSelected || isHighlighted ? cfg.color : `${cfg.color}55`}
                strokeWidth={isSelected ? 2.5 : 1.5}
              />
              <text
                y={1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="13"
                fill={cfg.color}
                className="pointer-events-none"
              >
                {n.node_type === 'dataset' ? '⬡' : n.node_type === 'column' ? '⬝' : n.node_type === 'model' ? '◈' : n.node_type === 'metric' ? '◇' : n.node_type === 'anomaly' ? '⚠' : n.node_type === 'forecast' ? '📈' : n.node_type === 'document' ? '📄' : '●'}
              </text>
              <text
                y={NODE_R + 12}
                textAnchor="middle"
                fontSize="8.5"
                fill="#CBD5E1"
                className="pointer-events-none"
              >
                {n.name.length > 18 ? n.name.slice(0, 17) + '…' : n.name}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main KnowledgeGraphPage
// ─────────────────────────────────────────────────────────────────────────────
export const KnowledgeGraphPage: React.FC = () => {
  // State
  const [projects, setProjects] = useState<Array<{ id: string; name: string }>>([]);
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string; target_column?: string | null }>>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [graphData, setGraphData] = useState<KnowledgeGraphData | null>(null);
  const [filteredTypes, setFilteredTypes] = useState<Set<string>>(new Set());
  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);
  const [scale, setScale] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [buildResult, setBuildResult] = useState<GraphBuildResult | null>(null);
  const [ragQuery, setRagQuery] = useState('');
  const [ragResult, setRagResult] = useState<GraphRAGResult | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [highlightIds, setHighlightIds] = useState<Set<string>>(new Set());
  const [activePanel, setActivePanel] = useState<'node' | 'rag' | 'metrics'>('metrics');

  // Load projects on mount
  useEffect(() => {
    api.getProjects?.().then(p => {
      if (Array.isArray(p)) setProjects(p);
      else if ((p as any)?.items) setProjects((p as any).items);
    }).catch(() => {});
  }, []);

  // Load datasets when project changes
  useEffect(() => {
    if (!selectedProjectId) return;
    api.getDatasets?.(selectedProjectId).then((d: any) => {
      const list = Array.isArray(d) ? d : d?.items || [];
      setDatasets(list);
      if (list.length > 0) setSelectedDatasetId(list[0].id);
    }).catch(() => {});

    // Try to load existing graph
    api.getKnowledgeGraph(selectedProjectId).then(g => {
      if (g.nodes.length > 0) setGraphData(g);
    }).catch(() => {});
  }, [selectedProjectId]);

  const handleBuild = async () => {
    if (!selectedProjectId) return;
    setLoading(true);
    try {
      const result = await api.buildKnowledgeGraph({
        project_id: selectedProjectId,
        dataset_id: selectedDatasetId || undefined,
        correlation_threshold: 0.25,
        include_models: true,
        include_anomalies: true,
        include_forecasts: true,
      });
      setBuildResult(result);
      const g = await api.getKnowledgeGraph(selectedProjectId);
      setGraphData(g);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleRagQuery = async () => {
    if (!ragQuery.trim()) return;
    setRagLoading(true);
    try {
      const result = await api.queryGraphRAG(ragQuery, selectedProjectId || undefined, 2);
      setRagResult(result);
      // Highlight seed entities in graph
      const ids = new Set<string>(result.seed_entities.map((e: any) => String(e.id)));
      setHighlightIds(ids);
      setActivePanel('rag');
    } catch (e) {
      console.error(e);
    } finally {
      setRagLoading(false);
    }
  };

  const handleNodeClick = (n: GraphNodeData) => {
    setSelectedNode(n);
    setHighlightIds(new Set([n.id]));
    setActivePanel('node');
  };

  // Filter nodes/edges by type
  const visibleNodes = graphData
    ? (filteredTypes.size === 0 ? graphData.nodes : graphData.nodes.filter(n => !filteredTypes.has(n.node_type)))
    : [];
  const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
  const visibleEdges = graphData
    ? graphData.edges.filter(e => visibleNodeIds.has(e.source_id) && visibleNodeIds.has(e.target_id))
    : [];

  const typeDistribution = graphData?.metrics?.node_type_distribution || {};
  const allTypes = Object.keys(typeDistribution);

  const nodeConfig = selectedNode ? getNodeConfig(selectedNode.node_type) : null;

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500/20 to-indigo-500/20 border border-violet-500/30 flex items-center justify-center">
            <Network className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Knowledge Graph & GraphRAG</h1>
            <p className="text-xs text-slate-400">Entity relationships, multi-hop relational intelligence & lineage tracing</p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-2">
          <select
            id="graph-project-select"
            value={selectedProjectId}
            onChange={e => setSelectedProjectId(e.target.value)}
            className="bg-slate-800/80 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-violet-500"
          >
            <option value="">— Select Project —</option>
            {projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          {datasets.length > 0 && (
            <select
              id="graph-dataset-select"
              value={selectedDatasetId}
              onChange={e => setSelectedDatasetId(e.target.value)}
              className="bg-slate-800/80 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-violet-500"
            >
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          )}
          <button
            id="build-graph-btn"
            onClick={handleBuild}
            disabled={!selectedProjectId || loading}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white text-xs font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <GitFork className="w-3.5 h-3.5" />}
            <span>{loading ? 'Building…' : 'Build Graph'}</span>
          </button>
        </div>
      </div>

      {/* Build success banner */}
      {buildResult && (
        <div className="flex items-center space-x-3 px-4 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 flex-shrink-0">
          <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>
            Graph built: <strong>{buildResult.total_nodes}</strong> nodes · <strong>{buildResult.total_edges}</strong> edges created in {buildResult.duration_ms.toFixed(0)} ms
          </span>
          <button onClick={() => setBuildResult(null)} className="ml-auto"><X className="w-3.5 h-3.5" /></button>
        </div>
      )}

      {/* GraphRAG Query Bar */}
      <div className="flex items-center space-x-2 flex-shrink-0">
        <div className="flex-1 relative">
          <Sparkles className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-violet-400" />
          <input
            id="graph-rag-input"
            type="text"
            value={ragQuery}
            onChange={e => setRagQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleRagQuery()}
            placeholder="Ask the Knowledge Graph — e.g. 'Which model achieved best F1 on churn?'"
            className="w-full bg-slate-800/60 border border-slate-700 focus:border-violet-500 text-slate-200 text-xs rounded-xl pl-9 pr-4 py-2.5 placeholder-slate-500 focus:outline-none transition-colors"
          />
        </div>
        <button
          id="graph-rag-btn"
          onClick={handleRagQuery}
          disabled={!ragQuery.trim() || ragLoading}
          className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-violet-600/80 hover:bg-violet-500 text-white text-xs font-semibold transition-all disabled:opacity-40"
        >
          {ragLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
          <span>{ragLoading ? 'Querying…' : 'GraphRAG'}</span>
        </button>
        {highlightIds.size > 0 && (
          <button
            onClick={() => { setHighlightIds(new Set()); setRagResult(null); }}
            className="px-3 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Main Content: Graph Canvas + Side Panel */}
      <div className="flex-1 flex space-x-4 min-h-0">

        {/* Graph Canvas */}
        <div className="flex-1 relative rounded-2xl overflow-hidden border border-slate-700/60 bg-slate-900/40 backdrop-blur-sm min-h-0">
          {/* Node type filter legend */}
          {allTypes.length > 0 && (
            <div className="absolute top-3 left-3 z-10 flex flex-wrap gap-1.5 max-w-xs">
              {allTypes.map(t => {
                const cfg = getNodeConfig(t);
                const isFiltered = filteredTypes.has(t);
                return (
                  <button
                    key={t}
                    onClick={() => setFilteredTypes(prev => {
                      const next = new Set(prev);
                      if (next.has(t)) next.delete(t); else next.add(t);
                      return next;
                    })}
                    className={`flex items-center space-x-1 px-2 py-1 rounded-full text-[10px] font-mono transition-all border ${
                      isFiltered ? 'opacity-30 border-slate-700' : 'border-opacity-40'
                    }`}
                    style={{ borderColor: cfg.color, color: cfg.color, backgroundColor: `${cfg.bg}cc` }}
                  >
                    <span>{t}</span>
                    <span className="opacity-60">({typeDistribution[t]})</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Zoom controls */}
          <div className="absolute bottom-4 right-4 z-10 flex flex-col space-y-1">
            <button onClick={() => setScale(s => Math.min(s + 0.15, 2.5))} className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center hover:bg-slate-700 transition-colors">
              <ZoomIn className="w-3.5 h-3.5 text-slate-300" />
            </button>
            <button onClick={() => setScale(s => Math.max(s - 0.15, 0.3))} className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center hover:bg-slate-700 transition-colors">
              <ZoomOut className="w-3.5 h-3.5 text-slate-300" />
            </button>
            <button onClick={() => setScale(1)} className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center hover:bg-slate-700 transition-colors">
              <RefreshCw className="w-3.5 h-3.5 text-slate-300" />
            </button>
          </div>

          {!graphData || visibleNodes.length === 0 ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 text-center px-8">
              <div className="w-16 h-16 rounded-full bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
                <Network className="w-8 h-8 text-violet-400/50" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-300">No Knowledge Graph yet</p>
                <p className="text-xs text-slate-500 mt-1">Select a project and click <strong>Build Graph</strong> to extract entities and relationships.</p>
              </div>
            </div>
          ) : (
            <div className="absolute inset-0">
              <GraphCanvas
                nodes={visibleNodes}
                edges={visibleEdges}
                selectedNode={selectedNode}
                highlightIds={highlightIds}
                onNodeClick={handleNodeClick}
                scale={scale}
              />
            </div>
          )}
        </div>

        {/* Side Panel */}
        <div className="w-72 flex flex-col space-y-3 flex-shrink-0">
          {/* Panel tabs */}
          <div className="flex rounded-xl bg-slate-800/60 border border-slate-700/60 p-1 text-xs">
            {(['metrics', 'node', 'rag'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActivePanel(tab)}
                className={`flex-1 py-1.5 rounded-lg capitalize font-medium transition-all ${
                  activePanel === tab
                    ? 'bg-violet-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab === 'rag' ? 'GraphRAG' : tab}
              </button>
            ))}
          </div>

          {/* Metrics panel */}
          {activePanel === 'metrics' && graphData && (
            <div className="flex-1 overflow-y-auto space-y-3">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Nodes', val: graphData.metrics.node_count },
                  { label: 'Edges', val: graphData.metrics.edge_count },
                  { label: 'Density', val: graphData.metrics.density },
                  { label: 'Connected', val: graphData.metrics.is_connected ? 'Yes' : 'No' },
                ].map(item => (
                  <div key={item.label} className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                    <div className="text-[10px] text-slate-400 font-mono">{item.label}</div>
                    <div className="text-sm font-bold text-white mt-0.5">{item.val}</div>
                  </div>
                ))}
              </div>

              {graphData.metrics.key_entities?.length > 0 && (
                <div className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                  <div className="text-[10px] text-slate-400 font-mono mb-2">KEY ENTITIES (PageRank)</div>
                  <div className="space-y-2">
                    {graphData.metrics.key_entities.slice(0, 5).map((e: any) => {
                      const cfg = getNodeConfig(e.type);
                      return (
                        <div key={e.node_id} className="flex items-center space-x-2">
                          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: cfg.color }} />
                          <span className="text-xs text-slate-300 flex-1 truncate">{e.name}</span>
                          <span className="text-[9px] font-mono text-slate-500">{e.score.toFixed(3)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {Object.keys(graphData.metrics.relation_distribution || {}).length > 0 && (
                <div className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                  <div className="text-[10px] text-slate-400 font-mono mb-2">RELATIONS</div>
                  <div className="space-y-1.5">
                    {Object.entries(graphData.metrics.relation_distribution)
                      .sort(([, a], [, b]) => (b as number) - (a as number))
                      .map(([rel, cnt]) => {
                        const color = RELATION_COLORS[rel] || '#94A3B8';
                        return (
                          <div key={rel} className="flex items-center space-x-2">
                            <div className="w-2 h-0.5 flex-shrink-0" style={{ backgroundColor: color }} />
                            <span className="text-[10px] text-slate-400 flex-1 truncate font-mono">{rel.replace(/_/g, ' ')}</span>
                            <span className="text-[9px] font-bold text-slate-300">{String(cnt)}</span>
                          </div>
                        );
                      })
                    }
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Node detail panel */}
          {activePanel === 'node' && selectedNode && nodeConfig && (
            <div className="flex-1 overflow-y-auto space-y-3">
              <div className="bg-slate-800/60 rounded-xl p-4 border" style={{ borderColor: `${nodeConfig.color}40` }}>
                <div className="flex items-center space-x-2 mb-3">
                  <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${nodeConfig.color}20` }}>
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: nodeConfig.color }} />
                  </div>
                  <div>
                    <p className="text-xs font-bold text-white">{selectedNode.name}</p>
                    <p className="text-[10px] font-mono mt-0.5" style={{ color: nodeConfig.color }}>
                      {selectedNode.node_type.toUpperCase()}
                    </p>
                  </div>
                </div>
                <div className="space-y-1.5">
                  <div className="text-[10px] text-slate-400 font-mono">NODE ID</div>
                  <code className="text-[9px] text-slate-400 bg-slate-900/60 px-2 py-1 rounded-lg block break-all">
                    {selectedNode.id}
                  </code>
                </div>
              </div>
              {Object.keys(selectedNode.properties).length > 0 && (
                <div className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                  <div className="text-[10px] text-slate-400 font-mono mb-2">PROPERTIES</div>
                  <div className="space-y-2">
                    {Object.entries(selectedNode.properties).slice(0, 10).map(([k, v]) => (
                      <div key={k} className="flex justify-between items-start">
                        <span className="text-[10px] text-slate-400 font-mono">{k}</span>
                        <span className="text-[10px] text-slate-200 ml-2 text-right break-all max-w-[120px]">
                          {typeof v === 'object' ? JSON.stringify(v).slice(0, 30) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* GraphRAG results panel */}
          {activePanel === 'rag' && ragResult && (
            <div className="flex-1 overflow-y-auto space-y-3">
              <div className={`flex items-center space-x-2 px-3 py-2 rounded-xl text-xs ${
                ragResult.grounded
                  ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
                  : 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-300'
              }`}>
                {ragResult.grounded
                  ? <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
                  : <XCircle className="w-3.5 h-3.5 flex-shrink-0" />}
                <span>{ragResult.grounded ? 'Grounded answer from graph' : 'No matching entities'}</span>
              </div>

              <div className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                <div className="text-[10px] text-slate-400 font-mono mb-2">ANSWER</div>
                <p className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">{ragResult.answer}</p>
              </div>

              {ragResult.facts.length > 0 && (
                <div className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                  <div className="text-[10px] text-slate-400 font-mono mb-2">EXTRACTED FACTS ({ragResult.facts.length})</div>
                  <div className="space-y-2">
                    {ragResult.facts.slice(0, 8).map((f, i) => (
                      <div key={i} className="flex items-start space-x-2">
                        <ChevronRight className="w-3 h-3 text-violet-400 flex-shrink-0 mt-0.5" />
                        <p className="text-[10px] text-slate-300">{f}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {ragResult.seed_entities.length > 0 && (
                <div className="bg-slate-800/60 rounded-xl p-3 border border-slate-700/60">
                  <div className="text-[10px] text-slate-400 font-mono mb-2">SEED ENTITIES</div>
                  <div className="space-y-1.5">
                    {ragResult.seed_entities.map((e: any, i: number) => {
                      const cfg = getNodeConfig(e.node_type);
                      return (
                        <div key={i} className="flex items-center space-x-2">
                          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: cfg.color }} />
                          <span className="text-[10px] text-slate-200 flex-1 truncate">{e.name}</span>
                          <span className="text-[9px] font-mono" style={{ color: cfg.color }}>{e.node_type}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="text-[10px] text-slate-500 font-mono text-right">
                {ragResult.subgraph.node_count} nodes · {ragResult.subgraph.edge_count} edges explored · {ragResult.duration_ms.toFixed(0)}ms
              </div>
            </div>
          )}

          {/* Empty panels */}
          {activePanel === 'node' && !selectedNode && (
            <div className="flex-1 flex flex-col items-center justify-center text-center space-y-2 opacity-50">
              <Info className="w-6 h-6 text-slate-400" />
              <p className="text-xs text-slate-400">Click any node in the graph to inspect its properties and relationships</p>
            </div>
          )}
          {activePanel === 'rag' && !ragResult && (
            <div className="flex-1 flex flex-col items-center justify-center text-center space-y-2 opacity-50">
              <Sparkles className="w-6 h-6 text-violet-400" />
              <p className="text-xs text-slate-400">Run a GraphRAG query above to see relational facts and lineage</p>
            </div>
          )}
          {activePanel === 'metrics' && !graphData && (
            <div className="flex-1 flex flex-col items-center justify-center text-center space-y-2 opacity-50">
              <BarChart2 className="w-6 h-6 text-slate-400" />
              <p className="text-xs text-slate-400">Build the knowledge graph to view topological metrics</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
