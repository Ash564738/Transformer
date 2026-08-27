// src/app/experiments/page.tsx
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getAuthToken, getBackendUrl } from "@/lib/api";

const BACKEND_PREFIX = getBackendUrl();

type Row = Record<string, unknown>;

interface ExperimentsReport {
  available?: boolean;
  reason?: string;
  metadata?: Row;
  executive_summary?: Row[];
  traditional_methods?: Row[];
  traditional_per_class?: Row[];
  traditional_combinations?: Row[];
  method_coverage?: Row[];
  method_gas_range?: Row[];
  supervised_ml?: Row[];
  weak_label_model?: Row[];
  weak_ml_transfer?: Row[];
  weak_transfer?: Row[];
  severity_records?: Row[];
  transformer_ranking?: Row[];
  ranking_stability?: Row[];
  cross_dataset_transfer?: Row[];
  rank_correlation_spearman?: Row[];
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card-surface mb-6 overflow-hidden">
      <div className="border-b border-cream-300 px-5 py-3">
        <h2 className="text-lg font-bold text-teal-900">{title}</h2>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function DataTable({
  rows,
  columns,
  maxRows = 50,
}: {
  rows: Row[];
  columns?: string[];
  maxRows?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  if (!rows?.length) {
    return (
      <p className="py-4 text-center text-sm text-teal-400">
        No data available.
      </p>
    );
  }

  const keys = columns?.length
    ? columns
    : Array.from(
        rows.reduce((set, row) => {
          Object.keys(row).forEach((key) => set.add(key));
          return set;
        }, new Set<string>())
      );

  const shown = expanded ? rows : rows.slice(0, maxRows);

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1000px] text-left text-sm">
          <thead>
            <tr className="border-b border-cream-300 bg-cream-50 text-xs font-semibold uppercase tracking-wide text-teal-400">
              {keys.map((key) => (
                <th key={key} className="px-3 py-2">
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, index) => (
              <tr
                key={index}
                className="border-b border-cream-200 hover:bg-cream-50"
              >
                {keys.map((key) => {
                  const value = row[key];
                  let display = "N/A";

                  if (value !== null && value !== undefined) {
                    if (typeof value === "number") {
                      display = Number.isFinite(value)
                        ? String(Math.round(value * 1000) / 1000)
                        : "N/A";
                    } else if (typeof value === "boolean") {
                      display = value ? "true" : "false";
                    } else if (typeof value === "object") {
                      display = JSON.stringify(value);
                    } else {
                      display = String(value);
                    }
                  }

                  return (
                    <td key={key} className="px-3 py-2 text-teal-700">
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {rows.length > maxRows && (
        <div className="mt-2 text-center">
          <button
            onClick={() => setExpanded((value) => !value)}
            className="rounded-lg border border-teal-200 px-3 py-1 text-xs font-semibold text-teal-600 hover:bg-cream-50"
          >
            {expanded ? "Show less" : `Show ${rows.length - maxRows} more rows`}
          </button>
        </div>
      )}
    </div>
  );
}

function MetadataCard({ metadata }: { metadata: Row }) {
  if (!metadata || !Object.keys(metadata).length) return null;

  return (
    <Section title="Experiment Metadata">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(metadata).map(([key, value]) => (
          <div key={key} className="rounded-lg bg-cream-50 p-3">
            <div className="text-xs font-semibold uppercase text-teal-400">
              {key}
            </div>
            <div className="mt-1 break-words text-sm text-teal-900">
              {typeof value === "object"
                ? JSON.stringify(value)
                : String(value)}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function MetricChart({
  rows,
  xKey,
  yKey,
  title,
  kind = "bar",
  height = 300,
}: {
  rows: Row[];
  xKey: string;
  yKey: string;
  title: string;
  kind?: "bar" | "line";
  height?: number;
}) {
  const data = rows
    .filter(
      (row) =>
        row[xKey] !== null &&
        row[xKey] !== undefined &&
        row[yKey] !== null &&
        row[yKey] !== undefined
    )
    .map((row) => ({
      [xKey]: String(row[xKey]),
      [yKey]: Number(row[yKey]),
    }));

  if (!data.length) return null;

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-teal-900">{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        {kind === "line" ? (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey={yKey}
              stroke="#c96f28"
              strokeWidth={2}
            />
          </LineChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey={xKey}
              interval={0}
              angle={-30}
              textAnchor="end"
              height={80}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey={yKey} fill="#184843" />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function RankingPriorityCard({ rows }: { rows: Row[] }) {
  if (!rows.length) return null;

  const top = rows[0];
  const status3 = rows.filter(
    (row) => Number(row.transformer_overall_severity_level) === 3
  ).length;
  const status2 = rows.filter(
    (row) => Number(row.transformer_overall_severity_level) === 2
  ).length;
  const status1 = rows.filter(
    (row) => Number(row.transformer_overall_severity_level) === 1
  ).length;

  return (
    <Section title="Fleet Maintenance Priority">
      <div className="rounded-xl border border-cream-300 bg-cream-50 p-5">
        <p className="text-sm text-teal-700">
          IEEE condition status and maintenance priority are separate. Ranking
          is an unweighted lexicographic evidence order. Concentration, delta,
          and rate evidence remain separate and are never summed into an
          invented health score.
        </p>

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="rounded-lg bg-white p-4">
            <div className="text-xs uppercase text-teal-400">Status 3</div>
            <div className="mt-1 text-2xl font-bold text-status-critical">
              {status3}
            </div>
          </div>
          <div className="rounded-lg bg-white p-4">
            <div className="text-xs uppercase text-teal-400">Status 2</div>
            <div className="mt-1 text-2xl font-bold text-teal-800">
              {status2}
            </div>
          </div>
          <div className="rounded-lg bg-white p-4">
            <div className="text-xs uppercase text-teal-400">Status 1</div>
            <div className="mt-1 text-2xl font-bold text-teal-800">
              {status1}
            </div>
          </div>
          <div className="rounded-lg bg-white p-4">
            <div className="text-xs uppercase text-teal-400">
              First priority
            </div>
            <div className="mt-1 text-xl font-bold text-teal-900">
              #{Number(top.rank ?? 1)} {String(top.transformer_id ?? "")}
            </div>
          </div>
        </div>

        <div className="mt-4 text-sm text-teal-700">
          <strong>Why first:</strong>{" "}
          {String(top.maintenance_priority_reason ?? "")} Current concentration
          exceedance ={" "}
          <strong>
            {String(
              top.current_concentration_exceedance_ratio ??
                top.current_standardized_exceedance ??
                "N/A"
            )}
            ×
          </strong>
          , rate exceedance ={" "}
          <strong>
            {String(top.current_rate_exceedance_ratio ?? "N/A")}×
          </strong>
          , delta exceedance ={" "}
          <strong>
            {String(top.current_delta_exceedance_ratio ?? "N/A")}×
          </strong>
          .
        </div>
      </div>
    </Section>
  );
}

export default function ExperimentsPage() {
  const [report, setReport] = useState<ExperimentsReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("summary");

  const fetchReport = useCallback(async () => {
    setLoading(true);
    setError(null);

    const token = getAuthToken();
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (token) headers.Authorization = `Bearer ${token}`;

    try {
      const response = await fetch(`${BACKEND_PREFIX}/report/experiments`, {
        headers,
        cache: "no-store",
      });

      const body = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof body.error === "string"
            ? body.error
            : "Failed to load experiments report."
        );
      }

      setReport(body as ExperimentsReport);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setReport(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReport();

    const handler = () => fetchReport();
    window.addEventListener("experiments-refresh", handler);

    return () => window.removeEventListener("experiments-refresh", handler);
  }, [fetchReport]);

  const tabs = useMemo(
    () => [
      ["summary", "Executive Summary"],
      ["traditional", "Traditional Methods"],
      ["combinations", "Combinations"],
      ["coverage", "Coverage / Range"],
      ["features", "Feature Ablation"],
      ["cross", "Cross-Dataset Transfer"],
      ["ml", "Supervised ML"],
      ["weak", "Weak Supervision"],
      ["severity", "Severity Records"],
      ["ranking", "Transformer Ranking"],
      ["stability", "History / Stability Evidence"],
      ["correlation", "Rank Correlation"],
    ],
    []
  );

  if (loading) {
    return <div className="p-8 text-center text-teal-500">Loading...</div>;
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <p className="text-status-critical">{error}</p>
        <button
          onClick={fetchReport}
          className="mt-4 rounded-lg border border-teal-200 px-4 py-2 text-sm font-semibold text-teal-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="p-8 text-center text-teal-400">
        No report data available.
      </div>
    );
  }

  if (report.available === false) {
    return (
      <div className="card-surface mx-auto max-w-3xl p-8 text-center">
        <h1 className="text-xl font-bold text-teal-900">
          No current experiment results
        </h1>
        <p className="mt-3 text-sm text-teal-600">
          {report.reason ??
            "The previous experiment artifacts are no longer available."}
        </p>
        <p className="mt-2 text-xs text-teal-400">
          Run the offline experiment pipeline again to publish a new research
          report.
        </p>
        <button
          onClick={fetchReport}
          className="mt-5 rounded-lg border border-teal-200 px-4 py-2 text-sm font-semibold text-teal-700"
        >
          Check again
        </button>
      </div>
    );
  }

  const ranking = report.transformer_ranking ?? [];
  const supervised = report.supervised_ml ?? [];

  return (
    <div className="space-y-6">
      <MetadataCard metadata={report.metadata ?? {}} />
      <RankingPriorityCard rows={ranking} />

      <div className="flex flex-wrap gap-2">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`rounded-lg px-4 py-2 text-sm font-semibold ${
              activeTab === id
                ? "bg-teal-900 text-white"
                : "bg-cream-50 text-teal-700 hover:bg-cream-100"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {activeTab === "summary" && (
        <>
          <Section title="Executive Summary">
            <DataTable rows={report.executive_summary ?? []} />
          </Section>
          <Section title="Traditional Diagnostic Benchmark">
            <MetricChart
              rows={report.traditional_methods ?? []}
              xKey="method"
              yKey="macro_f1"
              title="Traditional methods — Macro F1"
            />
          </Section>
          <Section title="Supervised ML — Locked Test">
            <MetricChart
              rows={supervised.filter((row) => row.split === "locked_test")}
              xKey="model"
              yKey="macro_f1"
              title="Supervised ML — Locked Test Macro F1"
            />
          </Section>
        </>
      )}

      {activeTab === "traditional" && (
        <>
          <Section title="Traditional Methods">
            <DataTable rows={report.traditional_methods ?? []} />
          </Section>
          <Section title="Per-Class Performance">
            <DataTable rows={report.traditional_per_class ?? []} />
          </Section>
        </>
      )}

      {activeTab === "combinations" && (
        <>
          <Section title="Traditional Combinations">
            <DataTable
              rows={report.traditional_combinations ?? []}
              maxRows={100}
            />
          </Section>
          <Section title="Best Locked-Test Combinations">
            <MetricChart
              rows={(report.traditional_combinations ?? [])
                .filter(
                  (row) =>
                    row.split === "locked_test" &&
                    row.granularity === "fine"
                )
                .sort(
                  (a, b) =>
                    Number(b.macro_f1 ?? 0) - Number(a.macro_f1 ?? 0)
                )
                .slice(0, 15)}
              xKey="methods"
              yKey="macro_f1"
              title="Top traditional combinations"
            />
          </Section>
        </>
      )}

      {activeTab === "coverage" && (
        <>
          <Section title="Method Coverage">
            <DataTable rows={report.method_coverage ?? []} />
          </Section>
          <Section title="Gas Range by Method">
            <DataTable rows={report.method_gas_range ?? []} />
          </Section>
        </>
      )}

      {activeTab === "features" && (
        <>
          <Section title="Feature Representation Comparison">
            <DataTable
              rows={supervised}
              columns={[
                "model",
                "feature_mode",
                "split",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
              ]}
              maxRows={100}
            />
          </Section>
          <MetricChart
            rows={supervised
              .filter((row) => row.split === "locked_test")
              .sort(
                (a, b) =>
                  Number(b.macro_f1 ?? 0) - Number(a.macro_f1 ?? 0)
              )}
            xKey="feature_mode"
            yKey="macro_f1"
            title="Locked-test Macro F1 by feature representation"
          />
        </>
      )}

      {activeTab === "cross" && (
        <>
          <Section title="Cross-Dataset Transfer">
            <DataTable
              rows={report.cross_dataset_transfer ?? []}
              columns={[
                "train_dataset",
                "test_dataset",
                "feature_mode",
                "model",
                "n_train",
                "n_test",
                "macro_f1",
                "accuracy",
                "balanced_accuracy",
              ]}
              maxRows={100}
            />
          </Section>
          <MetricChart
            rows={(report.cross_dataset_transfer ?? [])
              .sort(
                (a, b) =>
                  Number(b.macro_f1 ?? 0) - Number(a.macro_f1 ?? 0)
              )
              .slice(0, 20)}
            xKey="model"
            yKey="macro_f1"
            title="Cross-dataset transfer — top Macro F1 results"
          />
          <p className="px-1 text-xs text-teal-400">
            Cross-dataset transfer is evaluation evidence only and is not used
            to select the production weak student.
          </p>
        </>
      )}

      {activeTab === "ml" && (
        <>
          <Section title="Supervised ML Benchmark">
            <DataTable rows={supervised} maxRows={100} />
          </Section>
          <Section title="Locked-Test Comparison">
            <MetricChart
              rows={supervised.filter(
                (row) => row.split === "locked_test"
              )}
              xKey="model"
              yKey="macro_f1"
              title="Supervised models — locked test"
            />
          </Section>
        </>
      )}

      {activeTab === "weak" && (
        <>
          <Section title="Weak Label Model">
            <DataTable rows={report.weak_label_model ?? []} />
          </Section>
          <Section title="Weak ML Transfer">
            <DataTable
              rows={report.weak_ml_transfer ?? report.weak_transfer ?? []}
              maxRows={100}
            />
          </Section>
          <Section title="Weak ML Locked-Test Comparison">
            <MetricChart
              rows={(report.weak_ml_transfer ?? report.weak_transfer ?? [])
                .filter(
                  (row) =>
                    row.split === "locked_test" &&
                    row.granularity === "fine"
                )
                .sort(
                  (a, b) =>
                    Number(b.macro_f1 ?? 0) - Number(a.macro_f1 ?? 0)
                )
                .slice(0, 20)}
              xKey="model"
              yKey="macro_f1"
              title="Weak-supervision transfer — locked test"
            />
          </Section>
        </>
      )}

      {activeTab === "severity" && (
        <Section title="Severity Records">
          <DataTable rows={report.severity_records ?? []} maxRows={100} />
        </Section>
      )}

      {activeTab === "ranking" && (
        <>
          <Section title="Transformer Fleet Ranking">
            <DataTable
              rows={ranking}
              columns={[
                "rank",
                "transformer_id",
                "transformer_overall_severity_label",
                "current_concentration_exceedance_ratio",
                "current_delta_exceedance_ratio",
                "current_rate_exceedance_ratio",
                "current_standard_trigger_count",
                "table2_exceed_count",
                "table4_exceed_count",
                "table3_exceed_count",
                "history_max_status_before_current",
                "historical_max_standardized_exceedance",
                "history_current_fault_recurrence_fraction",
                "history_worsening_transition_ratio",
                "current_fault",
                "recommended_action",
              ]}
              maxRows={100}
            />
          </Section>

          <Section title="Top Transformer Priority">
            <MetricChart
              rows={ranking.slice(0, 20).map((row) => ({
                transformer_id: row.transformer_id,
                rank: Number(row.rank ?? 0),
              }))}
              xKey="transformer_id"
              yKey="rank"
              title="Top 20 fleet positions — lower rank = higher priority"
            />
          </Section>
        </>
      )}

      {activeTab === "stability" && (
        <Section title="History / Stability Evidence">
          <p className="mb-4 text-sm text-teal-600">
            Historical fields describe record count, observation span,
            recurrence, worsening/improving transitions, and history
            sufficiency. They are not converted into an arbitrary weighted
            health score.
          </p>
          <DataTable rows={report.ranking_stability ?? []} maxRows={100} />
        </Section>
      )}

      {activeTab === "correlation" && (
        <Section title="Rank Correlation — Descriptive Baselines">
          <p className="mb-4 text-sm text-teal-600">
            Spearman/Kendall correlations compare the documented fleet order
            with separate DGA evidence channels and TDCG. They are diagnostics
            only; no correlated baseline is fed back into ranking.
          </p>
          <DataTable
            rows={report.rank_correlation_spearman ?? []}
            columns={["metric_a", "metric_b", "n", "spearman_rho"]}
            maxRows={100}
          />
        </Section>
      )}
    </div>
  );
}