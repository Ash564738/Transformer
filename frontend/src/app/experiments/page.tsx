"use client";

import { useEffect, useState } from "react";
import { AuthGuard } from "@/components/auth/auth-guard";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area
} from "recharts";

// ─── Types ────────────────────────────────────────────────
interface AnomalyResult {
  "Precision@K": number;
  "Recall@K": number;
  Lift: number;
  GIC: number;
  Stability: number;
  Spearman: number;
  TC: number;
  Precision_CI_low: number;
  Precision_CI_high: number;
  Posthoc: Record<string, Record<string, number>>;
}

interface SupervisedMethodResult {
  [model: string]: { accuracy: number; f1_macro: number };
}

interface GasDistData {
  bin_centers: number[];
  counts: number[];
}

interface Section31 {
  gas_distributions: Record<string, GasDistData>;
  correlation_matrix: { columns: string[]; values: number[][] };
  missing_summary: { columns: string[]; missing_ratio: number[] };
}

interface Section32 {
  [method: string]: Record<string, number>;
}

interface Section34 {
  severity_histogram: GasDistData;
  severity_label_counts: Record<string, number>;
  top5_transformers: { transformer_id: string; severity_score: number }[];
}

// ─── Constants ────────────────────────────────────────────
const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8", "#82ca9d"];
const METHOD_DISPLAY_ORDER = ["tdcg", "iforest", "lof", "ocsvm", "autoencoder", "ensemble"];
const SUPERVISED_MODEL_ORDER = ["RF", "RF+SMOTE", "XGBoost", "XGBoost+SMOTE", "CatBoost", "CatBoost+SMOTE"];

export default function ExperimentReportPage() {
  const [token, setToken] = useState<string | null>(null);
  const [data, setData] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const storedToken = localStorage.getItem("dga-auth-token");
    setToken(storedToken);
  }, []);

  useEffect(() => {
    if (!token) { setLoading(false); return; }
    const apiUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5000"}/report/experiments`;
    fetch(apiUrl, { headers: { Authorization: `Bearer ${token}` } })
      .then(res => res.ok ? res.json() : Promise.reject(res.statusText))
      .then(json => setData(json))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [token]);

  if (loading) return <div className="p-8">Loading report data...</div>;
  if (error) return <div className="p-8 text-red-500">Error: {error}</div>;

  const anomaly: Record<string, AnomalyResult> = data.anomaly || {};
  const supervised: Record<string, SupervisedMethodResult> = data.supervised || {};
  const section31: Section31 | null = data.exploratory || null;
  const section32: Section32 | null = data.label_distribution || null;
  const section34: Section34 | null = data.risk_analysis || null;

  const gasDistToChart = (dist: GasDistData) =>
    dist.bin_centers.map((c, i) => ({ bin: c.toFixed(2), count: dist.counts[i] }));

  return (
    <AuthGuard>
      <div className="p-6 space-y-10 max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold text-teal-900">Experiment Data Overview</h1>

        {/* ─── 3.1 Exploratory Analysis ─── */}
        <section>
          <h2 className="text-lg font-semibold">Gas Distributions & Correlations</h2>
          {section31 && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                {Object.entries(section31.gas_distributions).map(([gas, dist]) => (
                  <div key={gas} className="border rounded p-2">
                    <p className="text-sm font-medium">{gas.toUpperCase()}</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={gasDistToChart(dist)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="bin" tick={false} />
                        <YAxis />
                        <Tooltip />
                        <Area type="monotone" dataKey="count" stroke="#0d9488" fill="#99f6e4" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>

              <h3 className="mt-6 font-medium">Correlation Matrix</h3>
              <div className="overflow-x-auto mt-2">
                <table className="min-w-full text-sm border">
                  <thead className="bg-teal-50">
                    <tr>
                      <th></th>
                      {section31.correlation_matrix.columns.map(c => <th key={c} className="px-2">{c}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {section31.correlation_matrix.values.map((row, i) => (
                      <tr key={i}>
                        <td className="font-medium">{section31.correlation_matrix.columns[i]}</td>
                        {row.map((val, j) => (
                          <td key={j} className="text-center" style={{ backgroundColor: val > 0.7 ? '#fee2e2' : val < -0.7 ? '#e0f2fe' : 'transparent' }}>
                            {val.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="mt-6 font-medium">Missing Data Summary</h3>
              <ResponsiveContainer width="100%" height={300} className="mt-2">
                <BarChart data={section31.missing_summary.columns.map((c, i) => ({ column: c, ratio: section31.missing_summary.missing_ratio[i] }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="column" />
                  <YAxis unit="%" />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Bar dataKey="ratio" fill="#f97316" />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </section>

        {/* ─── 3.2 Label Distribution ─── */}
        <section>
          <h2 className="text-lg font-semibold">Fault Label Distribution by Method</h2>
          {section32 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
              {Object.entries(section32).map(([method, counts]) => {
                const chartData = Object.entries(counts).map(([label, count]) => ({ label, count }));
                return (
                  <div key={method} className="border rounded p-2">
                    <p className="text-sm font-medium">{method}</p>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="label" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#0ea5e9" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* ─── 3.3 Supervised Model Performance ─── */}
        <section>
          <h2 className="text-lg font-semibold">Supervised Model Performance</h2>
          <div className="overflow-x-auto rounded-lg border border-teal-200 mt-4 mb-6">
            <table className="min-w-full text-sm">
              <thead className="bg-teal-50 text-teal-700">
                <tr>
                  <th className="px-4 py-2.5 text-left font-semibold">Labeling Method</th>
                  {SUPERVISED_MODEL_ORDER.map(model => <th key={model} className="px-4 py-2.5">{model}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-teal-100">
                {Object.entries(supervised).map(([method, models]) => (
                  <tr key={method}>
                    <td className="px-4 py-2.5 font-medium">{method}</td>
                    {SUPERVISED_MODEL_ORDER.map(model => {
                      const perf = models[model];
                      if (!perf) return <td key={model} className="px-4 py-2.5 text-center text-gray-400">–</td>;
                      return (
                        <td key={model} className="px-4 py-2.5 text-center">
                          <span className="font-medium">{(perf.accuracy * 100).toFixed(1)}%</span>
                          <span className="text-teal-500"> / </span>
                          <span>{(perf.f1_macro * 100).toFixed(1)}%</span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {Object.keys(supervised).length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium mb-2">Accuracy</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={Object.entries(supervised).flatMap(([method, models]) =>
                    Object.entries(models).map(([model, metrics]) => ({
                      method,
                      model,
                      value: metrics.accuracy * 100
                    }))
                  )}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis unit="%" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div>
                <h3 className="font-medium mb-2">Macro F1</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={Object.entries(supervised).flatMap(([method, models]) =>
                    Object.entries(models).map(([model, metrics]) => ({
                      method,
                      model,
                      value: metrics.f1_macro * 100
                    }))
                  )}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis unit="%" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </section>

        {/* ─── 3.4 Risk Score & Severity Analysis ─── */}
        <section>
          <h2 className="text-lg font-semibold">Risk Score & Severity Analysis</h2>
          {section34 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
              <div>
                <h3 className="font-medium mb-2">Severity Score Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={gasDistToChart(section34.severity_histogram)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="bin" tick={false} />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="count" stroke="#1e3a8a" fill="#bfdbfe" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div>
                <h3 className="font-medium mb-2">Severity Label Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={Object.entries(section34.severity_label_counts).map(([name, value]) => ({ name, value }))}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {Object.keys(section34.severity_label_counts).map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="md:col-span-2">
                <h3 className="font-medium mb-2">Top 5 Highest‑Risk Transformers</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={section34.top5_transformers}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="transformer_id" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="severity_score" fill="#dc2626" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </section>
      </div>
    </AuthGuard>
  );
}