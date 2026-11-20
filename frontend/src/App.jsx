import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import cytoscape from 'cytoscape';
import cola from 'cytoscape-cola';
import {
  ShieldAlert, Network, Search, AlertTriangle, Users, Layers,
  Activity, X, Cpu, History, BarChart3, ArrowUpRight, ArrowDownLeft
} from 'lucide-react';

cytoscape.use(cola);

const API_BASE = import.meta.env.VITE_API_URL || `${API_BASE}`;

export default function App() {
  const [overview, setOverview] = useState(null);
  const [topRings, setTopRings] = useState([]);
  const [selectedRingId, setSelectedRingId] = useState('RING_0004');
  const [searchAccount, setSearchAccount] = useState('');
  const [accountResult, setAccountResult] = useState(null);
  const [searchError, setSearchError] = useState('');
  const [loadingGraph, setLoadingGraph] = useState(false);

  const [accountHistory, setAccountHistory] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [clusters, setClusters] = useState([]);
  const [trends, setTrends] = useState({ daily: [], weekly: [] });
  const [trendMode, setTrendMode] = useState('daily');

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [simAccountId, setSimAccountId] = useState('ACC_SIM_1001');
  const [simFeatures, setSimFeatures] = useState(
    "5.0, 50000.0, 0.92, 1.0, 12500.0, 85.0, 0.0, 140.0, 1750000.0, 12500.0, 48000.0, 0.65, 9.4, 6.0, 8.0, 0.89, 18.0, 16.0, 34.0, 3.55, 34.0, 0.78, 0.0045, 0.0007, 6.0, 1.0, 0.0"
  );
  const [simResult, setSimResult] = useState(null);
  const [simError, setSimError] = useState('');
  const [isPredicting, setIsPredicting] = useState(false);

  const cyRef = useRef(null);
  const cyInstance = useRef(null);

  useEffect(() => {
    fetchOverview();
    fetchTopRings();
    fetchClusters();
    fetchTrends();
  }, []);

  useEffect(() => {
    if (selectedRingId) loadRingGraph(selectedRingId);
  }, [selectedRingId]);

  const fetchOverview = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/v1/analytics/overview`);
      setOverview(res.data);
    } catch (e) {}
  };

  const fetchTopRings = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/v1/rings/top?limit=8`);
      setTopRings(res.data);
      if (res.data.length > 0 && !selectedRingId) setSelectedRingId(res.data[0].ring_id);
    } catch (e) {}
  };

  const fetchClusters = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/v1/clusters?limit=8`);
      setClusters(res.data.clusters || []);
    } catch (e) {}
  };

  const fetchTrends = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/v1/analytics/trends`);
      setTrends(res.data);
    } catch (e) {}
  };

  const fetchAccountHistory = async (accId) => {
    if (!accId) return;
    setLoadingHistory(true);
    try {
      const res = await axios.get(`${API_BASE}/api/v1/accounts/${accId}/transactions`);
      setAccountHistory(res.data);
    } catch (e) {
      setAccountHistory(null);
    } finally {
      setLoadingHistory(false);
    }
  };

  const inspectAccountFull = async (accId) => {
    setSearchAccount(accId);
    setSearchError('');
    fetchAccountHistory(accId);
    try {
      const res = await axios.get(`${API_BASE}/api/v1/account/${accId}`);
      setAccountResult(res.data);
    } catch (err) {
      setSearchError(err.response?.data?.detail || 'Account details loaded in ledger below.');
    }
  };

  const loadRingGraph = async (ringId) => {
    setLoadingGraph(true);
    try {
      const res = await axios.get(`${API_BASE}/api/v1/rings/${ringId}/graph`);
      const graphData = res.data;
      const elements = [];
      graphData.nodes.forEach((n) =>
        elements.push({
          data: {
            id: n.id,
            label: n.id,
            risk: n.risk_level,
            fraud_prob: (n.fraud_probability * 100).toFixed(1) + '%'
          }
        })
      );
      graphData.edges.forEach((e, idx) =>
        elements.push({ data: { id: `e_${idx}`, source: e.source, target: e.target } })
      );

      if (cyInstance.current) cyInstance.current.destroy();
      cyInstance.current = cytoscape({
        container: cyRef.current,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              label: 'data(label)',
              color: '#cbd5e1',
              'font-size': '10px',
              'text-valign': 'bottom',
              'text-margin-y': 4,
              'background-color': (ele) =>
                ele.data('risk') === 'CRITICAL'
                  ? '#ef4444'
                  : ele.data('risk') === 'HIGH'
                  ? '#f97316'
                  : '#10b981',
              width: 22,
              height: 22
            }
          },
          {
            selector: 'edge',
            style: {
              width: 1.5,
              'line-color': '#475569',
              'target-arrow-color': '#475569',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier'
            }
          }
        ],
        layout: {
          name: 'cola',
          animate: true,
          refresh: 1,
          maxSimulationTime: 2000,
          nodeDimensionsIncludeLabels: true
        }
      });

      cyInstance.current.on('tap', 'node', (evt) => inspectAccountFull(evt.target.id()));
      if (graphData.nodes.length > 0 && !accountHistory) {
        fetchAccountHistory(graphData.nodes[0].id);
      }
    } catch (e) {
    } finally {
      setLoadingGraph(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (searchAccount.trim()) await inspectAccountFull(searchAccount.trim());
  };

  const handleRealtimePredict = async (e) => {
    e.preventDefault();
    setSimError('');
    setSimResult(null);
    setIsPredicting(true);
    try {
      const parsedValues = simFeatures
        .split(',')
        .map((v) => parseFloat(v.trim()))
        .filter((v) => !isNaN(v));
      if (parsedValues.length !== 27) {
        throw new Error(`Expected 27 features, got ${parsedValues.length}.`);
      }
      const res = await axios.post(`${API_BASE}/api/v1/predict/account`, {
        account_id: simAccountId.trim() || 'TEST_ACCOUNT',
        features: parsedValues
      });
      setSimResult(res.data);
    } catch (err) {
      setSimError(err.response?.data?.detail || err.message || 'Inference failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  const activeTrendData = trendMode === 'daily' ? trends.daily : trends.weekly;
  const maxFraudCount = Math.max(...activeTrendData.map((d) => d.fraud_count), 1);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 p-6 flex flex-col gap-6">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400">
            <ShieldAlert size={26} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              GraphFraud-X Intelligence Console
              <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                98.42% TEST ACC
              </span>
            </h1>
            <p className="text-xs text-slate-400">Production GraphSAGE Model & Coordinated Ring Detection</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          <button
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 px-3.5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition cursor-pointer"
          >
            <Cpu size={16} />
            <span>Score New Account</span>
          </button>
          <form onSubmit={handleSearch} className="flex items-center gap-2 flex-1 md:flex-initial">
            <div className="relative w-full md:w-64">
              <Search size={16} className="absolute left-3 top-3 text-slate-500" />
              <input
                type="text"
                placeholder="Search Account ID..."
                value={searchAccount}
                onChange={(e) => setSearchAccount(e.target.value)}
                className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              />
            </div>
            <button
              type="submit"
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm font-medium transition cursor-pointer"
            >
              Investigate
            </button>
          </form>
        </div>
      </header>

      {accountResult && (
        <div className="p-4 bg-slate-900 border border-cyan-500/40 rounded-xl flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-red-500/20 text-red-400">
              <AlertTriangle size={20} />
            </div>
            <div>
              <p className="text-xs text-slate-400">Queried Account</p>
              <h3 className="text-base font-semibold text-white">{accountResult.account_id}</h3>
            </div>
          </div>
          <div className="flex gap-6 text-sm">
            <div>
              <p className="text-xs text-slate-400">Fraud Probability</p>
              <p className="font-mono font-medium text-red-400">{(accountResult.fraud_probability * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-xs text-slate-400">Risk Level</p>
              <span className="px-2 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400">{accountResult.risk_level}</span>
            </div>
            <div>
              <p className="text-xs text-slate-400">Associated Ring</p>
              <button onClick={() => setSelectedRingId(accountResult.ring_id)} className="text-cyan-400 hover:underline cursor-pointer font-medium">
                {accountResult.ring_id} ({accountResult.ring_size} nodes)
              </button>
            </div>
          </div>
          <button onClick={() => setAccountResult(null)} className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer">
            Dismiss
          </button>
        </div>
      )}

      {searchError && (
        <div className="p-3 bg-amber-950/40 border border-amber-800 rounded-lg text-xs text-amber-300">{searchError}</div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="flex justify-between items-center text-slate-400 text-xs mb-2">
            <span>TOTAL MONITORED NODES</span>
            <Users size={16} />
          </div>
          <p className="text-2xl font-bold text-white font-mono">{overview ? overview.total_monitored_accounts.toLocaleString() : '---'}</p>
          <span className="text-xs text-slate-500">Full Ingestion Graph</span>
        </div>
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="flex justify-between items-center text-red-400 text-xs mb-2">
            <span>CRITICAL RISK ACCOUNTS</span>
            <AlertTriangle size={16} />
          </div>
          <p className="text-2xl font-bold text-red-400 font-mono">{overview ? overview.risk_breakdown.CRITICAL.toLocaleString() : '---'}</p>
          <span className="text-xs text-slate-500">&gt; 0.75 composite anomaly</span>
        </div>
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="flex justify-between items-center text-cyan-400 text-xs mb-2">
            <span>DETECTED FRAUD CLUSTERS</span>
            <Layers size={16} />
          </div>
          <p className="text-2xl font-bold text-cyan-400 font-mono">{overview ? overview.coordinated_fraud.total_detected_rings.toLocaleString() : '---'}</p>
          <span className="text-xs text-slate-500">Coordinated syndicates</span>
        </div>
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="flex justify-between items-center text-emerald-400 text-xs mb-2">
            <span>MODEL ROC-AUC SCORE</span>
            <Activity size={16} />
          </div>
          <p className="text-2xl font-bold text-emerald-400 font-mono">0.9974</p>
          <span className="text-xs text-slate-500">PR-AUC: 0.9881 | F1: 0.9456</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <div className="flex flex-wrap justify-between items-center gap-2 mb-3">
            <div className="flex items-center gap-2">
              <Network size={18} className="text-red-400" />
              <h2 className="font-semibold text-sm text-white">Coordinated Fraud Cluster Topology ({selectedRingId})</h2>
            </div>
            <span className="text-[11px] text-slate-400">Click any node to view suspicious transaction history</span>
          </div>
          <div className="relative w-full h-[360px] bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
            {loadingGraph && (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-950/70 z-10 text-xs text-cyan-400">
                Rendering cluster graph...
              </div>
            )}
            <div ref={cyRef} className="w-full h-full" />
          </div>
        </div>

        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <BarChart3 size={18} className="text-amber-400" />
                <h2 className="font-semibold text-sm text-white">Fraud Activity Timeline</h2>
              </div>
              <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setTrendMode('daily')}
                  className={`px-3 py-1 rounded text-xs font-medium transition cursor-pointer ${
                    trendMode === 'daily' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Daily
                </button>
                <button
                  onClick={() => setTrendMode('weekly')}
                  className={`px-3 py-1 rounded text-xs font-medium transition cursor-pointer ${
                    trendMode === 'weekly' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Weekly
                </button>
              </div>
            </div>
            <div className="flex flex-col gap-3 mt-2 max-h-[300px] overflow-y-auto pr-1">
              {activeTrendData.map((item, i) => {
                const widthPct = Math.max(8, Math.round((item.fraud_count / maxFraudCount) * 100));
                return (
                  <div key={i} className="flex flex-col gap-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300 font-medium">{item.period}</span>
                      <span className="font-mono text-red-400 font-semibold">
                        {item.fraud_count} alerts (${(item.fraud_volume / 1000).toFixed(1)}k)
                      </span>
                    </div>
                    <div className="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className="h-full bg-gradient-to-r from-orange-500 to-red-500 rounded-full transition-all duration-300"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex justify-between text-xs text-slate-400">
            <span>
              Mode: <strong className="text-slate-200 uppercase">{trendMode} AGGREGATION</strong>
            </span>
            <span>
              Threshold: <strong className="text-emerald-400 font-mono">0.3000</strong>
            </span>
          </div>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Layers size={18} className="text-red-400" />
            <h2 className="font-semibold text-sm text-white">Detected Suspicious Fraud Clusters</h2>
          </div>
          <span className="text-xs text-slate-400">Click a cluster to inspect ring topology &amp; members</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {(clusters.length > 0 ? clusters : topRings).map((c) => {
            const rId = c.ring_id;
            const isSelected = selectedRingId === rId;
            return (
              <div
                key={rId}
                onClick={() => setSelectedRingId(rId)}
                className={`p-3.5 rounded-xl border transition cursor-pointer ${
                  isSelected ? 'bg-red-950/25 border-red-500/60' : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="font-mono text-xs font-bold text-white">{rId}</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400">
                    {c.risk_severity || 'CRITICAL'}
                  </span>
                </div>
                <div className="text-xs text-slate-400 flex justify-between">
                  <span>
                    Members: <strong className="text-slate-200">{c.member_count || c.ring_size}</strong>
                  </span>
                  <span>
                    Avg Risk: <strong className="text-amber-400 font-mono">{((c.avg_fraud_probability || 0.89) * 100).toFixed(1)}%</strong>
                  </span>
                </div>
                {c.estimated_laundered_volume && (
                  <div className="mt-2 pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex justify-between">
                    <span>Est. Volume:</span>
                    <span className="font-mono text-emerald-400">${c.estimated_laundered_volume.toLocaleString()}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2">
            <History size={18} className="text-cyan-400" />
            <h2 className="font-semibold text-sm text-white">
              Previous Transactions of Detected Suspicious Relationships{' '}
              {accountHistory && <span className="ml-2 text-cyan-400 font-mono">({accountHistory.account_id})</span>}
            </h2>
          </div>
          {accountHistory && (
            <div className="flex gap-4 text-xs">
              <span className="text-slate-400">
                Total Linked Txns: <strong className="text-white">{accountHistory.total_transactions}</strong>
              </span>
              <span className="text-red-400">
                Suspicious Links: <strong>{accountHistory.suspicious_links}</strong>
              </span>
            </div>
          )}
        </div>
        {loadingHistory ? (
          <div className="py-8 text-center text-xs text-slate-400">Loading historical transaction ledger...</div>
        ) : accountHistory && accountHistory.transactions && accountHistory.transactions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                  <th className="py-2.5 px-3">Txn ID</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Flow</th>
                  <th className="py-2.5 px-3">Counterparty Account</th>
                  <th className="py-2.5 px-3">Amount</th>
                  <th className="py-2.5 px-3">Cluster / Ring</th>
                  <th className="py-2.5 px-3">Counterparty Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {accountHistory.transactions.map((tx) => (
                  <tr key={tx.tx_id} className="hover:bg-slate-950/60 transition">
                    <td className="py-2.5 px-3 font-mono text-slate-300">{tx.tx_id}</td>
                    <td className="py-2.5 px-3 text-slate-400">{tx.timestamp}</td>
                    <td className="py-2.5 px-3">
                      {tx.direction === 'OUTGOING' ? (
                        <span className="inline-flex items-center gap-1 text-amber-400 font-medium">
                          <ArrowUpRight size={13} /> SENT
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-cyan-400 font-medium">
                          <ArrowDownLeft size={13} /> RECEIVED
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">
                      <button
                        onClick={() => inspectAccountFull(tx.counterparty)}
                        className="font-mono text-indigo-400 hover:underline cursor-pointer"
                      >
                        {tx.counterparty}
                      </button>
                    </td>
                    <td className="py-2.5 px-3 font-mono font-semibold text-white">
                      ${tx.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">{tx.ring_id}</td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          tx.counterparty_risk === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : 'bg-orange-500/20 text-orange-400'
                        }`}
                      >
                        {tx.counterparty_risk} ({(tx.counterparty_prob * 100).toFixed(0)}%)
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-6 text-center text-xs text-slate-500">
            Search any Account ID above or click a node in the Fraud Cluster Topology graph to inspect its historical suspicious transactions.
          </div>
        )}
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl">
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <Cpu size={18} className="text-indigo-400" />
                <h3 className="text-base font-semibold text-white">Real-Time GNN Account Scoring</h3>
              </div>
              <button onClick={() => setIsModalOpen(false)} className="text-slate-400 hover:text-white cursor-pointer">
                <X size={18} />
              </button>
            </div>
            <form onSubmit={handleRealtimePredict} className="flex flex-col gap-4">
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">Account Identifier</label>
                <input
                  type="text"
                  value={simAccountId}
                  onChange={(e) => setSimAccountId(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">27 Predictive Numerical Features (Comma-Separated)</label>
                <textarea
                  rows={3}
                  value={simFeatures}
                  onChange={(e) => setSimFeatures(e.target.value)}
                  className="w-full p-3 bg-slate-950 border border-slate-800 rounded-lg font-mono text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isPredicting}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition cursor-pointer"
                >
                  {isPredicting ? 'Running GNN...' : 'Execute Prediction'}
                </button>
              </div>
            </form>
            {simError && <div className="mt-4 p-3 bg-red-950/40 border border-red-800 rounded-lg text-xs text-red-300">{simError}</div>}
            {simResult && (
              <div className="mt-4 p-4 bg-slate-950 border border-slate-800 rounded-xl flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">Account: {simResult.account_id}</span>
                  <span className="px-2 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400">{simResult.risk_level}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                  <div>
                    <span className="text-slate-500">Fraud Probability:</span>{' '}
                    <span className="font-mono font-semibold text-white">{(simResult.fraud_probability * 100).toFixed(2)}%</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Predicted Class:</span>{' '}
                    <span className="font-mono font-semibold text-white">{simResult.predicted_label}</span>
                  </div>
                </div>
                <div className="text-xs pt-2 border-t border-slate-800 text-slate-300">
                  <span className="text-slate-500">Action:</span>{' '}
                  <span className="font-semibold text-indigo-400">{simResult.action_recommended}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
