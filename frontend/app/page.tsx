"use client";

import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend
} from "recharts";
import { 
  Home, Activity, CheckCircle, SlidersHorizontal, BarChart3, 
  AlertCircle, X, Zap, Cpu, Search, BrainCircuit, ChevronDown, Trophy, 
  Download, FileText, Printer, FileArchive
} from "lucide-react";

interface Rule {
  rule: string;
  weight: number;
}
interface PredictionResponse {
  predictedPrice: number;
  topRules: Rule[];
  allRules?: Rule[];
  error?: string;
}
interface ModelMetrics {
  rmse: number;
  mae: number;
  r2: number;
}
interface MetricsResponse {
  models: Record<string, ModelMetrics>;
  error?: string;
}
interface MembershipFunction {
  mean: number;
  sigma: number;
}
interface FuzzySetsResponse {
  membershipFunctions: Record<string, MembershipFunction[]>;
  error?: string;
}
interface ChartPoint {
  x: string;
  MF1?: number;
  MF2?: number;
  MF3?: number;
  [key: string]: string | number | undefined;
}

export default function Dashboard() {
  const [medInc, setMedInc] = useState<number>(5.0);
  const [houseAge, setHouseAge] = useState<number>(25.0);
  const [aveRooms, setAveRooms] = useState<number>(5.0);

  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const [topRules, setTopRules] = useState<Rule[]>([]);
  const [allRules, setAllRules] = useState<Rule[]>([]);
  
  const [metrics, setMetrics] = useState<Record<string, ModelMetrics> | null>(null);
  const [fuzzySets, setFuzzySets] = useState<Record<string, MembershipFunction[]> | null>(null);
  
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState(0);
  
  // Modals
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isRuleModalOpen, setIsRuleModalOpen] = useState(false);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  const fetchMetricsAndFuzzySets = async () => {
    setIsLoading(true);
    try {
      const res1 = await axios.get<MetricsResponse>("http://127.0.0.1:8000/metrics");
      if (res1.data.error) setError(res1.data.error);
      else if (res1.data.models) setMetrics(res1.data.models);
      
      const res2 = await axios.get<FuzzySetsResponse>("http://127.0.0.1:8000/fuzzy-sets");
      if (res2.data.error && !error) setError(res2.data.error);
      else if (res2.data.membershipFunctions) setFuzzySets(res2.data.membershipFunctions);
      
      setIsLoading(false);
    } catch (err: unknown) {
      setError("Failed to connect to backend. Ensure it is running on port 8000.");
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    try {
      const res = await axios.post<PredictionResponse>("http://127.0.0.1:8000/predict", {
        MedInc: medInc,
        HouseAge: houseAge,
        AveRooms: aveRooms,
      });
      if (res.data.error) {
        setError(res.data.error);
        return;
      }
      setPredictedPrice(res.data.predictedPrice);
      setTopRules(res.data.topRules);
      if (res.data.allRules) {
        setAllRules(res.data.allRules);
      }
      setError("");
    } catch (err: unknown) {
      setError("Prediction request failed.");
    }
  };

  useEffect(() => {
    fetchMetricsAndFuzzySets();
  }, []);

  useEffect(() => {
    if (!isLoading && !error) {
      handlePredict();
    }
  }, [medInc, houseAge, aveRooms, isLoading, error]);
  
  const applyPreset = (inc: number, age: number, rooms: number) => {
    setMedInc(inc);
    setHouseAge(age);
    setAveRooms(rooms);
  };

  const getMFChartData = (featureIndex: number): ChartPoint[] => {
    if (!fuzzySets) return [];
    
    const featureKeys = Object.keys(fuzzySets);
    if (featureKeys.length <= featureIndex) return [];
    
    const featureKey = featureKeys[featureIndex];
    const mfs = fuzzySets[featureKey];
    if (!mfs) return [];
    
    const data: ChartPoint[] = [];
    for (let x = 0; x <= 1; x += 0.05) {
      const point: ChartPoint = { x: x.toFixed(2) };
      for (let i = 0; i < mfs.length; i++) {
        const { mean, sigma } = mfs[i];
        const val = Math.exp(-Math.pow(x - mean, 2) / (2 * Math.pow(sigma, 2)));
        point[`MF${i + 1}`] = val;
      }
      data.push(point);
    }
    return data;
  };

  const decodeRuleString = (ruleString: string, minimal: boolean = false) => {
    const match = ruleString.match(/Rule (\d+)/);
    if (!match) return "";
    const R = parseInt(match[1], 10);
    const mfNames = ["Low", "Medium", "High"];
    
    const c = R % 3;
    const b = Math.floor(R / 3) % 3;
    const a = Math.floor(R / 9) % 3;
    
    if (minimal) {
      return `MedInc:${mfNames[a]} | Age:${mfNames[b]} | Rooms:${mfNames[c]}`;
    }

    return (
       <div className="font-mono text-sm leading-loose">
         <span className="text-pink-500">IF</span> <span className="text-slate-300">Median Income</span> == <span className="text-cyan-400">"{mfNames[a]}"</span> <br/>
         <span className="text-pink-500 ml-4">AND</span> <span className="text-slate-300">House Age</span> == <span className="text-cyan-400">"{mfNames[b]}"</span> <br/>
         <span className="text-pink-500 ml-4">AND</span> <span className="text-slate-300">Average Rooms</span> == <span className="text-cyan-400">"{mfNames[c]}"</span> <br/>
         <span className="text-emerald-400">THEN</span> <span className="text-purple-400">evaluate(FirstOrderSugeno)</span>
       </div>
    );
  };

  const formatCurrency = (val: number): string => {
    return "$" + (val * 100000).toLocaleString(undefined, { maximumFractionDigits: 0 });
  };

  // Heuristic Sensitivity Calculation
  const maxInc = 15, maxAge = 52, maxRoom = 15;
  const wInc = (medInc / maxInc) * 1.5;
  const wAge = (houseAge / maxAge) * 1.0;
  const wRoom = (aveRooms / maxRoom) * 1.2;
  const wTotal = wInc + wAge + wRoom || 1;
  const pctInc = Math.round((wInc / wTotal) * 100);
  const pctAge = Math.round((wAge / wTotal) * 100);
  const pctRoom = Math.round((wRoom / wTotal) * 100);

  return (
    <>
      <div className="ambient-bg print:hidden">
        <div className="grid-mesh"></div>
      </div>

      <div className="min-h-screen text-slate-100 font-sans relative z-10 selection:bg-cyan-500/30 print:hidden">
        
        {/* Navbar */}
        <nav className="fixed top-0 w-full z-40 border-b border-white/10 bg-[#030712]/80 backdrop-blur-md">
          <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Activity className="w-6 h-6 text-cyan-400" />
              <span className="font-bold tracking-widest uppercase text-sm text-slate-200">ANFIS Predictor</span>
            </div>
            <div className="flex items-center gap-3">
               <button 
                 onClick={() => setIsExportModalOpen(true)}
                 className="flex items-center gap-2 px-4 py-2 bg-black/40 hover:bg-white/5 border border-purple-500/30 rounded-lg text-xs font-semibold text-purple-400 hover:text-purple-300 transition-all duration-300 shadow-[0_0_10px_rgba(168,85,247,0.1)]"
               >
                 <Download className="w-4 h-4" />
                 Report
               </button>
               <button 
                 onClick={() => setIsModalOpen(true)}
                 className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-xs font-semibold text-cyan-400 hover:text-cyan-300 transition-all duration-300 shadow-[0_0_10px_rgba(0,245,255,0.1)]"
               >
                 <Cpu className="w-4 h-4" />
                 Architecture
               </button>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="pt-32 pb-20 px-6 min-h-[90vh] flex flex-col items-center justify-center text-center max-w-5xl mx-auto">
          <div className="mb-6 inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-300 text-xs font-semibold uppercase tracking-wider shadow-[0_0_15px_rgba(168,85,247,0.2)]">
             <Zap className="w-3 h-3" />
             Soft Computing • Takagi-Sugeno
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold mb-6 tracking-tight">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-emerald-400 pb-2 inline-block">
              Neuro-Fuzzy
            </span>
            <br />
            Real Estate Intelligence
          </h1>
          
          <p className="text-lg md:text-xl text-slate-400 max-w-3xl mb-12 leading-relaxed">
            Bridging the interpretability of fuzzy logic with the self-optimizing power of deep neural architectures. 
            Experience non-linear valuation with complete rule transparency.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full mb-16">
            <div className="glass-card p-6 rounded-2xl flex flex-col items-center gap-3">
               <div className="w-12 h-12 rounded-full bg-cyan-500/10 flex items-center justify-center border border-cyan-500/30 shadow-[0_0_15px_rgba(0,245,255,0.2)]">
                  <Cpu className="w-5 h-5 text-cyan-400" />
               </div>
               <h3 className="font-semibold text-slate-200">5-Layer Architecture</h3>
               <p className="text-xs text-slate-400 text-center">Adaptive neural network natively tuning fuzzy inference parameters.</p>
            </div>
            <div className="glass-card p-6 rounded-2xl flex flex-col items-center gap-3">
               <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                  <Search className="w-5 h-5 text-emerald-400" />
               </div>
               <h3 className="font-semibold text-slate-200">100% Transparent</h3>
               <p className="text-xs text-slate-400 text-center">Interpretable rule base uncovers the exact reasoning behind every prediction.</p>
            </div>
            <div className="glass-card p-6 rounded-2xl flex flex-col items-center gap-3">
               <div className="w-12 h-12 rounded-full bg-purple-500/10 flex items-center justify-center border border-purple-500/30 shadow-[0_0_15px_rgba(168,85,247,0.2)]">
                  <BrainCircuit className="w-5 h-5 text-purple-400" />
               </div>
               <h3 className="font-semibold text-slate-200">Hybrid Tuning</h3>
               <p className="text-xs text-slate-400 text-center">Gradient descent backpropagation combined with least-squares estimation.</p>
            </div>
          </div>
          
          <button 
             onClick={() => document.getElementById('valuation-engine')?.scrollIntoView({ behavior: 'smooth' })}
             className="group relative px-8 py-4 bg-cyan-500 text-slate-900 font-bold rounded-xl overflow-hidden transition-transform hover:scale-105 active:scale-95 shadow-[0_0_30px_rgba(0,245,255,0.4)] hover:shadow-[0_0_50px_rgba(0,245,255,0.6)]"
          >
             <span className="relative z-10 flex items-center gap-2">
               Launch Predictor
               <ChevronDown className="w-5 h-5 group-hover:translate-y-1 transition-transform" />
             </span>
             <div className="absolute inset-0 bg-white/20 -translate-x-full group-hover:translate-x-0 transition-transform duration-500 skew-x-12"></div>
          </button>
        </section>

        {/* Dashboard Section */}
        <section id="valuation-engine" className="max-w-7xl mx-auto px-6 pb-32 pt-20">
          
          {error && (
            <div className="mb-8 p-4 glass-card border-red-500/50 bg-red-500/10 rounded-xl text-red-200 flex items-center gap-3">
              <AlertCircle className="w-6 h-6 text-red-400 shrink-0" />
              <p>{error}</p>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column */}
            <div className="flex flex-col gap-8">
              
              {/* Valuation Card */}
              <div className="glass-card rounded-2xl p-8 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-400 via-purple-500 to-emerald-400" />
                <h2 className="text-xl font-bold mb-6 flex items-center gap-2 tracking-wide text-slate-200">
                  <Home className="w-5 h-5 text-cyan-400" />
                  VALUATION
                </h2>
                
                {/* Presets */}
                <div className="flex flex-wrap gap-2 mb-6">
                   <button onClick={() => applyPreset(2.5, 35, 4.2)} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs text-slate-300 hover:border-cyan-500/50 transition-colors">
                      Budget Suburban
                   </button>
                   <button onClick={() => applyPreset(5.8, 18, 6.0)} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs text-slate-300 hover:border-purple-500/50 transition-colors">
                      Midtown Family
                   </button>
                   <button onClick={() => applyPreset(11.2, 5, 9.5)} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs text-slate-300 hover:border-emerald-500/50 transition-colors">
                      Executive Luxury
                   </button>
                </div>

                {isLoading ? (
                  <div className="animate-pulse flex flex-col gap-2 mb-10">
                      <div className="h-14 bg-white/10 rounded w-2/3"></div>
                      <div className="h-4 bg-white/5 rounded w-1/3"></div>
                  </div>
                ) : (
                  <div className="mb-10">
                    <div className="text-5xl lg:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 mb-2 tracking-tighter text-glow">
                      {predictedPrice !== null ? formatCurrency(predictedPrice) : "---"}
                    </div>
                    
                    {/* Feature Sensitivity Breakdown */}
                    <div className="mt-4">
                      <div className="flex gap-1 w-full h-1.5 rounded-full overflow-hidden bg-white/5">
                        <div style={{ width: `${pctInc}%` }} className="bg-cyan-400 shadow-[0_0_10px_#00f5ff]" title={`Income Impact: ${pctInc}%`} />
                        <div style={{ width: `${pctAge}%` }} className="bg-purple-500 shadow-[0_0_10px_#a855f7]" title={`Age Factor: ${pctAge}%`} />
                        <div style={{ width: `${pctRoom}%` }} className="bg-emerald-400 shadow-[0_0_10px_#10b981]" title={`Room Factor: ${pctRoom}%`} />
                      </div>
                      <div className="flex justify-between text-[10px] text-slate-500 mt-2 font-mono uppercase tracking-widest">
                        <span className="text-cyan-400/80 font-bold">Inc {pctInc}%</span>
                        <span className="text-purple-400/80 font-bold">Age {pctAge}%</span>
                        <span className="text-emerald-400/80 font-bold">Room {pctRoom}%</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="space-y-8">
                  <div>
                    <label className="flex justify-between text-xs uppercase tracking-wider font-semibold mb-3 text-slate-400">
                      <span>Median Income</span>
                      <span className="font-mono text-cyan-300">{medInc.toFixed(1)}</span>
                    </label>
                    <input
                      type="range" min="0.5" max="15" step="0.1"
                      value={medInc} onChange={(e) => setMedInc(parseFloat(e.target.value))}
                      disabled={isLoading}
                      className="w-full neon-slider disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <label className="flex justify-between text-xs uppercase tracking-wider font-semibold mb-3 text-slate-400">
                      <span>House Age (Years)</span>
                      <span className="font-mono text-purple-300">{houseAge.toFixed(0)}</span>
                    </label>
                    <input
                      type="range" min="1" max="52" step="1"
                      value={houseAge} onChange={(e) => setHouseAge(parseFloat(e.target.value))}
                      disabled={isLoading}
                      className="w-full neon-slider disabled:opacity-50"
                      style={{ background: 'rgba(255,255,255,0.1)' }}
                    />
                  </div>

                  <div>
                    <label className="flex justify-between text-xs uppercase tracking-wider font-semibold mb-3 text-slate-400">
                      <span>Average Rooms</span>
                      <span className="font-mono text-emerald-300">{aveRooms.toFixed(1)}</span>
                    </label>
                    <input
                      type="range" min="1" max="15" step="0.1"
                      value={aveRooms} onChange={(e) => setAveRooms(parseFloat(e.target.value))}
                      disabled={isLoading}
                      className="w-full neon-slider disabled:opacity-50"
                    />
                  </div>
                </div>
              </div>
              
              {/* Benchmarks Card */}
              <div className="glass-card rounded-2xl p-6">
                <h2 className="text-lg font-bold mb-6 flex items-center gap-2 tracking-wide text-slate-200 uppercase text-sm">
                  <BarChart3 className="w-4 h-4 text-purple-400" />
                  Benchmarks
                </h2>
                {isLoading ? (
                  <div className="space-y-4 animate-pulse">
                      <div className="h-12 bg-white/5 rounded-xl w-full"></div>
                      <div className="h-12 bg-white/5 rounded-xl w-full"></div>
                      <div className="h-12 bg-white/5 rounded-xl w-full"></div>
                  </div>
                ) : metrics ? (
                  <div className="space-y-3">
                    {Object.keys(metrics).map((modelName) => {
                        const isAnfis = modelName.toUpperCase() === "ANFIS";
                        return (
                          <div 
                            key={modelName} 
                            className={`p-3 rounded-xl border flex flex-col gap-2 relative overflow-hidden ${
                                isAnfis 
                                  ? 'bg-purple-500/10 border-purple-500/50 shadow-[0_0_15px_rgba(168,85,247,0.15)]' 
                                  : 'bg-white/5 border-white/5'
                            }`}
                          >
                            {isAnfis && <div className="absolute top-0 right-0 px-2 py-0.5 bg-purple-500 text-[10px] font-bold text-white rounded-bl-lg">BEST</div>}
                            <span className={`font-bold text-sm ${isAnfis ? 'text-purple-300' : 'text-slate-300'}`}>{modelName}</span>
                            <div className="flex gap-4 text-xs font-mono">
                                <span className="text-slate-400"><span className="text-slate-500 mr-1">RMSE:</span>{metrics[modelName].rmse?.toFixed(3)}</span>
                                <span className="text-slate-400"><span className="text-slate-500 mr-1">MAE:</span>{metrics[modelName].mae?.toFixed(3)}</span>
                                <span className="text-slate-400"><span className="text-slate-500 mr-1">R²:</span>{metrics[modelName].r2?.toFixed(3)}</span>
                            </div>
                          </div>
                        );
                    })}
                  </div>
                ) : (
                  <div className="text-slate-500 italic text-sm">Metrics not available.</div>
                )}
              </div>
            </div>

            {/* Right Column */}
            <div className="lg:col-span-2 flex flex-col gap-8">
              
              {/* Active Rules Card */}
              <div className="glass-card rounded-2xl p-6 flex flex-col h-full">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-lg font-bold flex items-center gap-2 tracking-wide text-slate-200 uppercase text-sm">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    Active Rule Firings
                  </h2>
                  <button onClick={() => setIsRuleModalOpen(true)} className="px-3 py-1 bg-white/5 border border-white/10 hover:border-emerald-500/50 rounded text-xs font-semibold text-emerald-400 transition-colors">
                    Inspect Full Rule Base (27 Rules)
                  </button>
                </div>
                
                <svg style={{ height: 0, width: 0, position: 'absolute' }}>
                  <defs>
                    <linearGradient id="neonGradient" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#00f5ff" />
                      <stop offset="100%" stopColor="#a855f7" />
                    </linearGradient>
                  </defs>
                </svg>

                {isLoading ? (
                    <div className="h-64 w-full bg-white/5 rounded-xl animate-pulse"></div>
                ) : topRules.length > 0 ? (
                    <>
                      <div className="h-64 w-full mb-6">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={topRules} layout="vertical" margin={{ left: 40, right: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" horizontal={false} />
                            <XAxis type="number" stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={{ stroke: '#ffffff20' }} />
                            <YAxis dataKey="rule" type="category" stroke="#64748b" tick={{ fill: '#94a3b8' }} axisLine={{ stroke: '#ffffff20' }} />
                            <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#030712', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} itemStyle={{ color: '#00f5ff' }} />
                            <Bar dataKey="weight" fill="url(#neonGradient)" radius={[0, 4, 4, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      
                      {/* Terminal Interpretable Rule */}
                      <div className="mt-auto bg-black/60 rounded-xl border border-cyan-500/30 p-5 shadow-[inset_0_0_20px_rgba(0,245,255,0.05)] relative overflow-hidden group">
                        <div className="absolute top-0 left-0 w-1 h-full bg-cyan-500 shadow-[0_0_10px_#00f5ff]"></div>
                        <h3 className="text-[10px] uppercase tracking-[0.2em] font-bold text-cyan-500 mb-4 flex items-center gap-2">
                           <Trophy className="w-3 h-3" />
                           Dominant Node Path
                        </h3>
                        {decodeRuleString(topRules[0].rule)}
                      </div>
                    </>
                ) : (
                    <div className="text-slate-500 italic h-64 flex items-center justify-center">Awaiting inference...</div>
                )}
              </div>

              {/* Membership Functions Card */}
              <div className="glass-card rounded-2xl p-6">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
                    <h2 className="text-lg font-bold flex items-center gap-2 tracking-wide text-slate-200 uppercase text-sm">
                    <SlidersHorizontal className="w-4 h-4 text-pink-400" />
                    Feature Membership
                    </h2>
                    
                    <div className="flex bg-black/40 rounded-full p-1 border border-white/10">
                      {["MedInc", "House Age", "Ave Rooms"].map((feat, idx) => (
                          <button
                            key={feat}
                            onClick={() => setSelectedFeatureIdx(idx)}
                            className={`px-4 py-1.5 text-xs font-bold rounded-full transition-all duration-300 ${
                                selectedFeatureIdx === idx 
                                    ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-[0_0_10px_rgba(0,245,255,0.4)]" 
                                    : "text-slate-400 hover:text-white hover:bg-white/5"
                            }`}
                          >
                            {feat}
                          </button>
                      ))}
                    </div>
                </div>

                <div className="h-72 w-full">
                    {isLoading ? (
                      <div className="h-full w-full bg-white/5 rounded-xl animate-pulse"></div>
                    ) : fuzzySets ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={getMFChartData(selectedFeatureIdx)} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                          <XAxis dataKey="x" stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={{ stroke: '#ffffff20' }} />
                          <YAxis stroke="#64748b" tick={{ fill: '#64748b' }} axisLine={{ stroke: '#ffffff20' }} />
                          <Tooltip contentStyle={{ backgroundColor: '#030712', borderColor: '#334155', borderRadius: '8px' }} />
                          <Legend wrapperStyle={{ paddingTop: '20px' }} />
                          <Line type="monotone" name="Low (MF1)" dataKey="MF1" stroke="#00f5ff" dot={false} strokeWidth={3} style={{ filter: 'drop-shadow(0px 0px 5px rgba(0,245,255,0.6))' }} />
                          <Line type="monotone" name="Medium (MF2)" dataKey="MF2" stroke="#a855f7" dot={false} strokeWidth={3} style={{ filter: 'drop-shadow(0px 0px 5px rgba(168,85,247,0.6))' }} />
                          <Line type="monotone" name="High (MF3)" dataKey="MF3" stroke="#10b981" dot={false} strokeWidth={3} style={{ filter: 'drop-shadow(0px 0px 5px rgba(16,185,129,0.6))' }} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="text-slate-500 italic flex items-center justify-center h-full">Loading clusters...</div>
                    )}
                </div>
              </div>

            </div>
          </div>
        </section>
      </div>

      {/* ANFIS Architecture Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 print:hidden">
           <div className="absolute inset-0 bg-black/80 backdrop-blur-md" onClick={() => setIsModalOpen(false)}></div>
           <div className="glass-card border border-cyan-500/30 rounded-2xl shadow-[0_0_50px_rgba(0,245,255,0.1)] max-w-3xl w-full max-h-[85vh] flex flex-col relative z-10 overflow-hidden transform animate-in zoom-in-95 duration-200">
              <div className="bg-black/40 p-6 border-b border-white/10 flex justify-between items-center z-10">
                 <h2 className="text-2xl font-bold flex items-center gap-3 text-white tracking-tight">
                    <Cpu className="w-6 h-6 text-cyan-400" />
                    System Architecture
                 </h2>
                 <button onClick={() => setIsModalOpen(false)} className="text-slate-400 hover:text-white p-2 rounded-full hover:bg-white/10 transition-colors">
                    <X className="w-6 h-6" />
                 </button>
              </div>
              <div className="p-8 space-y-6 overflow-y-auto custom-scrollbar bg-gradient-to-b from-transparent to-black/40">
                 <div className="group relative p-5 bg-black/40 rounded-xl border border-white/5 hover:border-cyan-500/30 transition-colors">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur"></div>
                    <h3 className="text-lg font-bold text-cyan-400 mb-2 flex items-center gap-2"><span className="text-white/30 text-sm">L1</span> Fuzzification</h3>
                    <p className="text-sm text-slate-400 mb-4 leading-relaxed">Projects inputs into the fuzzy space using differentiable Gaussian membership functions.</p>
                    <code className="block bg-black/60 p-4 rounded-lg font-mono text-sm text-pink-400 border border-white/5">
                       μ(x) = exp( - (x - c)² / (2σ²) )
                    </code>
                 </div>
                 <div className="group relative p-5 bg-black/40 rounded-xl border border-white/5 hover:border-purple-500/30 transition-colors">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur"></div>
                    <h3 className="text-lg font-bold text-purple-400 mb-2 flex items-center gap-2"><span className="text-white/30 text-sm">L2</span> Rule Firing Strengths</h3>
                    <p className="text-sm text-slate-400 mb-4 leading-relaxed">Applies the Product T-Norm operator to evaluate the antecedent of each rule.</p>
                    <code className="block bg-black/60 p-4 rounded-lg font-mono text-sm text-purple-300 border border-white/5">
                       w_i = μ_A(x) × μ_B(y) × μ_C(z)
                    </code>
                 </div>
                 <div className="group relative p-5 bg-black/40 rounded-xl border border-white/5 hover:border-blue-500/30 transition-colors">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur"></div>
                    <h3 className="text-lg font-bold text-blue-400 mb-2 flex items-center gap-2"><span className="text-white/30 text-sm">L3</span> Normalization</h3>
                    <p className="text-sm text-slate-400 mb-4 leading-relaxed">Converts raw firing strengths into normalized probability distributions across the rule base.</p>
                    <code className="block bg-black/60 p-4 rounded-lg font-mono text-sm text-cyan-300 border border-white/5">
                       w̄_i = w_i / Σ(w_k)
                    </code>
                 </div>
                 <div className="group relative p-5 bg-black/40 rounded-xl border border-white/5 hover:border-emerald-500/30 transition-colors">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur"></div>
                    <h3 className="text-lg font-bold text-emerald-400 mb-2 flex items-center gap-2"><span className="text-white/30 text-sm">L4</span> Sugeno Consequents</h3>
                    <p className="text-sm text-slate-400 mb-4 leading-relaxed">Computes local linear models tailored to each fuzzy region of the input space.</p>
                    <code className="block bg-black/60 p-4 rounded-lg font-mono text-sm text-emerald-300 border border-white/5">
                       f_i = p_i·x + q_i·y + r_i·z + s_i
                    </code>
                 </div>
                 <div className="group relative p-5 bg-black/40 rounded-xl border border-white/5 hover:border-amber-500/30 transition-colors">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-amber-500 to-orange-500 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur"></div>
                    <h3 className="text-lg font-bold text-amber-400 mb-2 flex items-center gap-2"><span className="text-white/30 text-sm">L5</span> Defuzzification</h3>
                    <p className="text-sm text-slate-400 mb-4 leading-relaxed">Aggregates all local models into a single, global crisp prediction scalar.</p>
                    <code className="block bg-black/60 p-4 rounded-lg font-mono text-sm text-amber-300 border border-white/5">
                       Output = Σ(w̄_i × f_i)
                    </code>
                 </div>
              </div>
           </div>
        </div>
      )}

      {/* Rule Explorer Modal */}
      {isRuleModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 print:hidden">
           <div className="absolute inset-0 bg-black/80 backdrop-blur-md" onClick={() => setIsRuleModalOpen(false)}></div>
           <div className="glass-card border border-emerald-500/30 rounded-2xl shadow-[0_0_50px_rgba(16,185,129,0.1)] max-w-4xl w-full max-h-[85vh] flex flex-col relative z-10 overflow-hidden transform animate-in zoom-in-95 duration-200">
              <div className="bg-black/40 p-6 border-b border-white/10 flex justify-between items-center z-10">
                 <h2 className="text-xl font-bold flex items-center gap-3 text-white tracking-tight">
                    <FileArchive className="w-5 h-5 text-emerald-400" />
                    Complete 27-Rule Base Explorer
                 </h2>
                 <button onClick={() => setIsRuleModalOpen(false)} className="text-slate-400 hover:text-white p-2 rounded-full hover:bg-white/10 transition-colors">
                    <X className="w-6 h-6" />
                 </button>
              </div>
              <div className="p-6 overflow-y-auto custom-scrollbar bg-black/20">
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {allRules.map((rule) => {
                     const isFiring = rule.weight > 0.05;
                     return (
                        <div key={rule.rule} className={`p-4 rounded-xl border transition-all ${isFiring ? 'bg-emerald-900/20 border-emerald-500/60 shadow-[0_0_15px_rgba(16,185,129,0.2)]' : 'bg-white/5 border-white/5 opacity-60'}`}>
                           <div className="flex justify-between items-center mb-2 border-b border-white/10 pb-2">
                              <span className={`font-bold text-sm ${isFiring ? 'text-emerald-400' : 'text-slate-400'}`}>{rule.rule}</span>
                              <span className="font-mono text-xs text-slate-400">w: {rule.weight.toFixed(4)}</span>
                           </div>
                           <div className="text-xs text-slate-300 font-mono tracking-tight leading-relaxed">
                              {decodeRuleString(rule.rule, true)}
                           </div>
                        </div>
                     )
                  })}
                </div>
              </div>
           </div>
        </div>
      )}

      {/* Export Summary Modal (Printable) */}
      {isExportModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
           <div className="absolute inset-0 bg-black/80 backdrop-blur-md print:hidden" onClick={() => setIsExportModalOpen(false)}></div>
           <div className="glass-card print:border-none print:shadow-none print:bg-white print:text-black border border-purple-500/30 rounded-2xl shadow-[0_0_50px_rgba(168,85,247,0.1)] max-w-2xl w-full max-h-[95vh] flex flex-col relative z-10 overflow-hidden transform animate-in zoom-in-95 duration-200">
              <div className="bg-black/40 print:bg-white p-6 border-b border-white/10 print:border-slate-300 flex justify-between items-center z-10">
                 <h2 className="text-xl font-bold flex items-center gap-3 text-white print:text-black tracking-tight">
                    <FileText className="w-5 h-5 text-purple-400 print:text-black" />
                    Valuation Summary Report
                 </h2>
                 <div className="flex items-center gap-2 print:hidden">
                    <button onClick={() => window.print()} className="flex items-center gap-2 px-3 py-1.5 bg-purple-500 hover:bg-purple-400 text-white rounded text-xs font-bold transition-colors">
                       <Printer className="w-4 h-4" /> Print
                    </button>
                    <button onClick={() => setIsExportModalOpen(false)} className="text-slate-400 hover:text-white p-1 rounded-full hover:bg-white/10 transition-colors">
                       <X className="w-6 h-6" />
                    </button>
                 </div>
              </div>
              <div className="p-8 space-y-8 overflow-y-auto custom-scrollbar bg-black/20 print:bg-white print:text-black">
                 <div className="text-center pb-6 border-b border-white/10 print:border-slate-300">
                    <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 print:text-black print:bg-none mb-2">
                       {predictedPrice !== null ? formatCurrency(predictedPrice) : "---"}
                    </h1>
                    <p className="text-sm text-slate-400 print:text-slate-600">ANFIS Estimated Market Value</p>
                 </div>
                 
                 <div>
                    <h3 className="text-sm font-bold text-slate-300 print:text-black uppercase tracking-wider mb-4">Property Vector</h3>
                    <div className="grid grid-cols-3 gap-4">
                       <div className="bg-white/5 print:bg-slate-100 p-4 rounded-lg border border-white/5 print:border-slate-300">
                          <p className="text-xs text-slate-400 print:text-slate-500 mb-1">Median Income</p>
                          <p className="font-mono text-lg text-cyan-300 print:text-black">{medInc}</p>
                       </div>
                       <div className="bg-white/5 print:bg-slate-100 p-4 rounded-lg border border-white/5 print:border-slate-300">
                          <p className="text-xs text-slate-400 print:text-slate-500 mb-1">House Age</p>
                          <p className="font-mono text-lg text-purple-300 print:text-black">{houseAge} yrs</p>
                       </div>
                       <div className="bg-white/5 print:bg-slate-100 p-4 rounded-lg border border-white/5 print:border-slate-300">
                          <p className="text-xs text-slate-400 print:text-slate-500 mb-1">Ave Rooms</p>
                          <p className="font-mono text-lg text-emerald-300 print:text-black">{aveRooms}</p>
                       </div>
                    </div>
                 </div>

                 <div>
                    <h3 className="text-sm font-bold text-slate-300 print:text-black uppercase tracking-wider mb-4">Dominant Logical Rule</h3>
                    <div className="bg-black/40 print:bg-slate-100 p-5 rounded-lg border border-cyan-500/30 print:border-slate-300">
                       {topRules.length > 0 ? decodeRuleString(topRules[0].rule, true) : "N/A"}
                    </div>
                 </div>

                 <div>
                    <h3 className="text-sm font-bold text-slate-300 print:text-black uppercase tracking-wider mb-4">Model Reliability (ANFIS)</h3>
                    <div className="bg-white/5 print:bg-slate-100 p-5 rounded-lg border border-white/5 print:border-slate-300 flex justify-around">
                       <div className="text-center">
                          <p className="text-xs text-slate-400 print:text-slate-500">RMSE</p>
                          <p className="font-mono text-slate-200 print:text-black">{metrics?.ANFIS?.rmse?.toFixed(3) || "---"}</p>
                       </div>
                       <div className="text-center">
                          <p className="text-xs text-slate-400 print:text-slate-500">MAE</p>
                          <p className="font-mono text-slate-200 print:text-black">{metrics?.ANFIS?.mae?.toFixed(3) || "---"}</p>
                       </div>
                       <div className="text-center">
                          <p className="text-xs text-slate-400 print:text-slate-500">R² Score</p>
                          <p className="font-mono text-slate-200 print:text-black">{metrics?.ANFIS?.r2?.toFixed(3) || "---"}</p>
                       </div>
                    </div>
                 </div>
                 
                 <div className="pt-6 border-t border-white/10 print:border-slate-300 text-center text-xs text-slate-500 print:text-slate-400">
                    Generated by Neuro-Fuzzy Real Estate Intelligence
                 </div>
              </div>
           </div>
        </div>
      )}

    </>
  );
}
