import React, { useState, useEffect, useRef } from 'react';
import { 
  Network, 
  UploadCloud, 
  FileText, 
  Search, 
  Send, 
  Trash2, 
  CheckCircle2, 
  AlertTriangle, 
  Sparkles, 
  BookOpen, 
  ExternalLink,
  Layers,
  ChevronRight,
  Hash,
  ShieldCheck,
  RefreshCw
} from 'lucide-react';
import { api, ragApi } from '../services/api';
import { Project, RAGDocument, RAGQueryResponse, HybridSearchResultItem } from '../types';

export const RAGPage: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(false);

  // Upload State
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Q&A State
  const [query, setQuery] = useState<string>('');
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [ragResult, setRagResult] = useState<RAGQueryResponse | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);

  // Hybrid Search Inspector State
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<HybridSearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [topK, setTopK] = useState<number>(5);

  // Active view tab
  const [activeTab, setActiveTab] = useState<'qa' | 'documents' | 'search'>('qa');

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const projs = await api.getProjects();
        setProjects(projs);
        if (projs.length > 0) {
          setSelectedProjectId(projs[0].id);
        }
      } catch (err) {
        console.error('Failed to fetch projects:', err);
      }
    };
    fetchProjects();
  }, []);

  useEffect(() => {
    if (!selectedProjectId) return;
    loadDocuments();
  }, [selectedProjectId]);

  const loadDocuments = async () => {
    setIsLoadingDocs(true);
    try {
      const docs = await ragApi.listDocuments(selectedProjectId);
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setIsLoadingDocs(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    setIsUploading(true);
    try {
      await ragApi.uploadDocument(file, selectedProjectId);
      await loadDocuments();
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setUploadError(err?.response?.data?.detail || 'Failed to index document.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    try {
      await ragApi.deleteDocument(docId);
      setDocuments(prev => prev.filter(d => d.document_id !== docId));
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  const handleRunQuery = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setIsQuerying(true);
    try {
      const res = await ragApi.queryRAG(query, selectedProjectId, 4);
      setRagResult(res);
      if (res.citations.length > 0) {
        setSelectedCitationId(res.citations[0].citation_id);
      }
    } catch (err: any) {
      console.error('Query failed:', err);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleRunSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const res = await ragApi.hybridSearch(searchQuery, selectedProjectId, topK);
      setSearchResults(res.results);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/30">
              Phase 6 Active
            </span>
            <span className="text-xs text-slate-400 font-mono">Dense Vector + BM25 Hybrid Retrieval</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-1 flex items-center space-x-3">
            <Network className="w-7 h-7 text-nexus-accent" />
            <span>Document Intelligence & Production RAG</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Ingest enterprise documents, chunk with semantic overlap, and execute grounded Q&A with verifiable evidence citations.
          </p>
        </div>

        {/* Project Selector & Upload CTA */}
        <div className="flex items-center space-x-3">
          <select
            value={selectedProjectId}
            onChange={(e) => setSelectedProjectId(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-nexus-accent"
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.md,.csv"
            className="hidden"
            onChange={handleFileUpload}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-nexus-accent to-nexus-purple text-white shadow-glow flex items-center space-x-2 hover:opacity-95 transition-all"
          >
            {isUploading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Chunking & Indexing...</span>
              </>
            ) : (
              <>
                <UploadCloud className="w-4 h-4" />
                <span>Upload Document</span>
              </>
            )}
          </button>
        </div>
      </div>

      {uploadError && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center space-x-3">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('qa')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all ${
            activeTab === 'qa'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Grounded Q&A Console
        </button>
        <button
          onClick={() => setActiveTab('documents')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all flex items-center space-x-2 ${
            activeTab === 'documents'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <BookOpen className="w-3.5 h-3.5" />
          <span>Ingested Documents ({documents.length})</span>
        </button>
        <button
          onClick={() => setActiveTab('search')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all flex items-center space-x-2 ${
            activeTab === 'search'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Search className="w-3.5 h-3.5" />
          <span>Hybrid Search Inspector</span>
        </button>
      </div>

      {/* TAB 1: Grounded Q&A Console */}
      {activeTab === 'qa' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            {/* Search Input Box */}
            <form onSubmit={handleRunQuery} className="glass-panel p-4 rounded-2xl border-slate-800 space-y-3">
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  placeholder="Ask a question grounded in your uploaded project documentation..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-nexus-accent"
                />
                <button
                  type="submit"
                  disabled={isQuerying || !query.trim()}
                  className="px-5 py-2.5 rounded-xl text-xs font-semibold bg-nexus-accent text-white hover:bg-nexus-accent/80 transition-all flex items-center space-x-2 disabled:opacity-50"
                >
                  {isQuerying ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  <span>Ask RAG</span>
                </button>
              </div>

              {/* Sample Quick Questions */}
              <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] text-slate-400">
                <span className="font-mono">Quick prompts:</span>
                {[
                  'What is the data architecture specification?',
                  'How does NEXUS prevent LLM hallucination?',
                  'What are the cross-validation guidelines?'
                ].map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setQuery(q);
                    }}
                    className="px-2 py-0.5 rounded-md bg-slate-800/80 hover:bg-slate-800 text-slate-300 text-[10px]"
                  >
                    "{q}"
                  </button>
                ))}
              </div>
            </form>

            {/* Answer Display */}
            {ragResult ? (
              <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                      ragResult.grounded ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                    }`}>
                      {ragResult.grounded ? 'GROUNDED EVIDENCE' : 'BASELINE CONTEXT'}
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      Query executed in {ragResult.duration_ms.toFixed(1)}ms
                    </span>
                  </div>
                  <span className="text-xs font-mono text-slate-400">
                    {ragResult.citations.length} Verified Citations
                  </span>
                </div>

                <div className="text-xs text-slate-200 leading-relaxed space-y-3 whitespace-pre-wrap">
                  {ragResult.answer}
                </div>
              </div>
            ) : (
              <div className="glass-panel p-12 rounded-2xl border-slate-800 text-center text-slate-400 text-xs space-y-2">
                <Search className="w-8 h-8 mx-auto text-slate-600" />
                <p>Ask a question above to retrieve context-grounded answers backed by document citations.</p>
              </div>
            )}
          </div>

          {/* Citations Panel */}
          <div className="lg:col-span-1 space-y-3">
            <h3 className="text-xs font-mono uppercase text-slate-400 font-semibold px-1 flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-pink-400" />
              <span>Evidence Provenance</span>
            </h3>

            {ragResult && ragResult.citations.length > 0 ? (
              <div className="space-y-3">
                {ragResult.citations.map((cite) => (
                  <div
                    key={cite.citation_id}
                    onClick={() => setSelectedCitationId(cite.citation_id)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer ${
                      selectedCitationId === cite.citation_id
                        ? 'bg-slate-800/90 border-nexus-accent shadow-glow'
                        : 'bg-slate-900/60 border-slate-800 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono mb-1.5">
                      <span className="text-nexus-accent font-bold">{cite.citation_tag}</span>
                      <span className="text-emerald-400">{(cite.confidence * 100).toFixed(1)}% match</span>
                    </div>
                    <div className="text-xs font-semibold text-white">{cite.label}</div>
                    <p className="text-[11px] text-slate-400 italic mt-2 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                      "{cite.snippet}"
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="glass-panel p-8 rounded-2xl border-slate-800 text-center text-slate-500 text-xs">
                No active citations. Submit a question to inspect verifiable source provenance.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: Ingested Documents */}
      {activeTab === 'documents' && (
        <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white">Indexed Knowledge Documents</h2>
            <span className="text-xs font-mono text-slate-400">{documents.length} Total Files</span>
          </div>

          {documents.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase text-[10px]">
                  <tr>
                    <th className="py-2.5 px-3">Filename</th>
                    <th className="py-2.5 px-3">Format</th>
                    <th className="py-2.5 px-3">Size</th>
                    <th className="py-2.5 px-3">Pages</th>
                    <th className="py-2.5 px-3">Chunks</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  {documents.map((doc) => (
                    <tr key={doc.document_id} className="hover:bg-slate-800/40">
                      <td className="py-3 px-3 flex items-center space-x-2 font-sans font-medium text-white">
                        <FileText className="w-4 h-4 text-nexus-accent" />
                        <span>{doc.filename}</span>
                      </td>
                      <td className="py-3 px-3 uppercase text-slate-400">{doc.file_type}</td>
                      <td className="py-3 px-3 text-slate-400">{(doc.file_size / 1024).toFixed(1)} KB</td>
                      <td className="py-3 px-3 text-slate-300">{doc.page_count}</td>
                      <td className="py-3 px-3 text-nexus-accent">{doc.chunk_count}</td>
                      <td className="py-3 px-3">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 uppercase">
                          {doc.status}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <button
                          onClick={() => handleDeleteDocument(doc.document_id)}
                          className="p-1.5 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-400 text-xs">
              No documents indexed for this project yet. Use "Upload Document" to index your first file.
            </div>
          )}
        </div>
      )}

      {/* TAB 3: Hybrid Search Inspector */}
      {activeTab === 'search' && (
        <div className="space-y-6">
          <form onSubmit={handleRunSearch} className="glass-panel p-5 rounded-2xl border-slate-800 space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="text"
                placeholder="Query dense embeddings + BM25 keyword index..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-nexus-accent"
              />
              <button
                type="submit"
                disabled={isSearching || !searchQuery.trim()}
                className="px-5 py-2.5 rounded-xl text-xs font-semibold bg-nexus-accent text-white hover:bg-nexus-accent/80 transition-all flex items-center space-x-2"
              >
                {isSearching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                <span>Execute Search</span>
              </button>
            </div>

            <div className="flex items-center space-x-4 text-xs text-slate-400">
              <label className="font-mono">Top K Candidates:</label>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="accent-nexus-accent"
              />
              <span className="font-mono text-white">{topK}</span>
            </div>
          </form>

          {searchResults.length > 0 ? (
            <div className="space-y-3">
              <h3 className="text-xs font-mono uppercase text-slate-400 font-semibold px-1">Ranked Candidates</h3>
              {searchResults.map((item, idx) => (
                <div key={idx} className="glass-panel p-4 rounded-xl border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-nexus-accent font-bold">Rank #{idx + 1}</span>
                    <span className="text-slate-400">Score: {item.score ? item.score.toFixed(4) : 'N/A'}</span>
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed">{item.text}</p>
                  {item.metadata && (
                    <div className="text-[10px] font-mono text-slate-400 flex items-center space-x-3 pt-1 border-t border-slate-800/80">
                      <span>Doc: {item.metadata.filename || item.document_id}</span>
                      <span>Page: {item.metadata.page || 1}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="glass-panel p-12 rounded-2xl border-slate-800 text-center text-slate-400 text-xs">
              Execute a hybrid query above to inspect cross-entropy reranking candidates.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
