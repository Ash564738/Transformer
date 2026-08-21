// src/app/experiments/page.tsx
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { getAuthToken } from "@/lib/api";
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

const BACKEND_PREFIX =
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  "http://127.0.0.1:5000";

interface ExperimentsReport {
  metadata?: Record<string, unknown>;
  executive_summary?: Array<Record<string, unknown>>;
  traditional_methods?: Array<Record<string, unknown>>;
  traditional_per_class?: Array<Record<string, unknown>>;
  traditional_combinations?: Array<Record<string, unknown>>;
  method_coverage?: Array<Record<string, unknown>>;
  method_gas_range?: Array<Record<string, unknown>>;
  supervised_ml?: Array<Record<string, unknown>>;
  weak_label_model?: Array<Record<string, unknown>>;
  weak_ml_transfer?: Array<Record<string, unknown>>;
  severity_records?: Array<Record<string, unknown>>;
  transformer_ranking?: Array<Record<string, unknown>>;
  ranking_stability?: Array<Record<string, unknown>>;
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
        <h2 className="text-lg font-bold text-teal-900">
          {title}
        </h2>
      </div>

      <div className="p-4">
        {children}
      </div>
    </div>
  );
}

function DataTable({
  rows,
  columns,
  maxRows = 50,
}: {
  rows: Array<Record<string, unknown>>;
  columns?: string[];
  maxRows?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  if (!rows || rows.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-teal-400">
        No data available.
      </p>
    );
  }

  const keys =
    columns && columns.length > 0
      ? columns
      : Array.from(
          rows.reduce<Set<string>>(
            (acc, row) => {
              Object.keys(row).forEach((key) =>
                acc.add(key)
              );
              return acc;
            },
            new Set<string>()
          )
        );

  const shownRows = expanded
    ? rows
    : rows.slice(0, maxRows);

  const hasMore = rows.length > maxRows;

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-left text-sm">
          <thead>
            <tr className="border-b border-cream-300 bg-cream-50 text-xs font-semibold uppercase tracking-wide text-teal-400">
              {keys.map((key) => (
                <th
                  key={key}
                  className="px-3 py-2"
                >
                  {key}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {shownRows.map((row, idx) => (
              <tr
                key={idx}
                className="border-b border-cream-200 hover:bg-cream-50"
              >
                {keys.map((key) => {
                  const value = row[key];

                  let display: string;

                  if (
                    value === null ||
                    value === undefined
                  ) {
                    display = "N/A";
                  } else if (
                    typeof value === "number"
                  ) {
                    display =
                      Number.isFinite(value)
                        ? String(
                            Math.round(
                              value * 1000
                            ) / 1000
                          )
                        : "N/A";
                  } else if (
                    typeof value === "boolean"
                  ) {
                    display = value
                      ? "true"
                      : "false";
                  } else if (
                    typeof value === "object"
                  ) {
                    display = JSON.stringify(
                      value
                    );
                  } else {
                    display = String(value);
                  }

                  return (
                    <td
                      key={key}
                      className="px-3 py-2 text-teal-700"
                    >
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {hasMore && (
        <div className="mt-2 text-center">
          <button
            onClick={() =>
              setExpanded(!expanded)
            }
            className="rounded-lg border border-teal-200 px-3 py-1 text-xs font-semibold text-teal-600 hover:bg-cream-50"
          >
            {expanded
              ? "Show less"
              : `Show ${
                  rows.length - maxRows
                } more rows`}
          </button>
        </div>
      )}
    </div>
  );
}

function MetadataCard({
  metadata,
}: {
  metadata: Record<string, unknown>;
}) {
  if (
    !metadata ||
    Object.keys(metadata).length === 0
  ) {
    return null;
  }

  return (
    <Section title="Experiment Metadata">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(metadata).map(
          ([key, value]) => (
            <div
              key={key}
              className="rounded-lg bg-cream-50 p-3"
            >
              <div className="text-xs font-semibold uppercase text-teal-400">
                {key}
              </div>

              <div className="mt-1 break-words text-sm text-teal-900">
                {typeof value ===
                "object"
                  ? JSON.stringify(value)
                  : String(value)}
              </div>
            </div>
          )
        )}
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
  rows: Array<Record<string, unknown>>;
  xKey: string;
  yKey: string;
  title: string;
  kind?: "bar" | "line";
  height?: number;
}) {
  if (!rows || rows.length === 0) {
    return null;
  }

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

  if (data.length === 0) {
    return null;
  }

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-teal-900">
        {title}
      </h3>

      <ResponsiveContainer
        width="100%"
        height={height}
      >
        {kind === "bar" ? (
          <BarChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e8e5d9"
              vertical={false}
            />
            <XAxis
              dataKey={xKey}
              tick={{
                fontSize: 11,
                fill: "#4f8f83",
              }}
              interval={0}
              angle={-30}
              textAnchor="end"
              height={80}
            />
            <YAxis
              tick={{
                fontSize: 11,
                fill: "#4f8f83",
              }}
            />
            <Tooltip />
            <Bar
              dataKey={yKey}
              fill="#184843"
            />
          </BarChart>
        ) : (
          <LineChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e8e5d9"
              vertical={false}
            />
            <XAxis
              dataKey={xKey}
              tick={{
                fontSize: 11,
                fill: "#4f8f83",
              }}
            />
            <YAxis
              tick={{
                fontSize: 11,
                fill: "#4f8f83",
              }}
            />
            <Tooltip />
            <Line
              type="monotone"
              dataKey={yKey}
              stroke="#c96f28"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

export default function ExperimentsPage() {
  const [report, setReport] =
    useState<ExperimentsReport | null>(
      null
    );

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);

  const [activeTab, setActiveTab] =
    useState("summary");

  const fetchReport =
    useCallback(async () => {
      setLoading(true);
      setError(null);

      const token = getAuthToken();

      const headers: Record<
        string,
        string
      > = {
        "Content-Type":
          "application/json",
      };

      if (token) {
        headers.Authorization =
          `Bearer ${token}`;
      }

      try {
        const res = await fetch(
          `${BACKEND_PREFIX}/report/experiments`,
          {
            headers,
            cache: "no-store",
          }
        );

        if (!res.ok) {
          const body =
            await res
              .json()
              .catch(() => ({
                error:
                  res.statusText,
              }));

          throw new Error(
            body.error ??
              "Failed to load experiments report."
          );
        }

        const data =
          await res.json();

        setReport(data);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unknown error"
        );
      } finally {
        setLoading(false);
      }
    }, []);

  useEffect(() => {
    fetchReport();

    const handler = () => {
      fetchReport();
    };

    window.addEventListener(
      "experiments-refresh",
      handler
    );

    return () => {
      window.removeEventListener(
        "experiments-refresh",
        handler
      );
    };
  }, [fetchReport]);

  const tabs = useMemo(
    () => [
      {
        id: "summary",
        label: "Executive Summary",
      },
      {
        id: "traditional",
        label: "Traditional Methods",
      },
      {
        id: "combinations",
        label: "Combinations",
      },
      {
        id: "coverage",
        label: "Coverage / Range",
      },
      {
        id: "ml",
        label: "Supervised ML",
      },
      {
        id: "weak",
        label: "Weak Supervision",
      },
      {
        id: "severity",
        label: "Severity Records",
      },
      {
        id: "ranking",
        label: "Transformer Ranking",
      },
      {
        id: "stability",
        label: "History / Stability Evidence",
      },
    ],
    []
  );

  if (loading) {
    return (
      <div className="p-8 text-center text-teal-500">
        Loading experiments report…
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <p className="text-status-critical">
          {error}
        </p>

        <button
          onClick={fetchReport}
          className="mt-4 rounded-lg border border-teal-200 px-4 py-2 text-sm font-semibold text-teal-700 hover:bg-cream-50"
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

  return (
    <div className="space-y-6">
      <MetadataCard
        metadata={
          report.metadata ?? {}
        }
      />

      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() =>
              setActiveTab(tab.id)
            }
            className={`rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
              activeTab === tab.id
                ? "bg-teal-900 text-white"
                : "bg-cream-50 text-teal-700 hover:bg-cream-100"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "summary" && (
        <>
          <Section title="Executive Summary">
            <DataTable
              rows={
                report.executive_summary ??
                []
              }
            />
          </Section>

          <Section title="Traditional Diagnostic Benchmark">
            <MetricChart
              rows={
                report.traditional_methods ??
                []
              }
              xKey="method"
              yKey="macro_f1"
              title="Traditional methods — Macro F1"
            />
          </Section>

          <Section title="ML Comparison">
            <MetricChart
              rows={
                (
                  report.supervised_ml ??
                  []
                ).filter(
                  (x) =>
                    x.split ===
                    "locked_test"
                )
              }
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
            <DataTable
              rows={
                report.traditional_methods ??
                []
              }
            />
          </Section>

          <Section title="Per-Class Performance">
            <DataTable
              rows={
                report.traditional_per_class ??
                []
              }
            />
          </Section>
        </>
      )}

      {activeTab === "combinations" && (
        <>
          <Section title="Traditional Combinations">
            <DataTable
              rows={
                report.traditional_combinations ??
                []
              }
              maxRows={100}
            />
          </Section>

          <Section title="Best Locked-Test Combinations">
            <MetricChart
              rows={(
                report.traditional_combinations ??
                []
              )
                .filter(
                  (r) =>
                    r.split ===
                      "locked_test" &&
                    r.granularity ===
                      "fine"
                )
                .sort(
                  (a, b) =>
                    Number(
                      b.macro_f1 ?? 0
                    ) -
                    Number(
                      a.macro_f1 ?? 0
                    )
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
            <DataTable
              rows={
                report.method_coverage ??
                []
              }
            />
          </Section>

          <Section title="Gas Range by Method">
            <DataTable
              rows={
                report.method_gas_range ??
                []
              }
            />
          </Section>
        </>
      )}

      {activeTab === "ml" && (
        <>
          <Section title="Supervised ML Benchmark">
            <DataTable
              rows={
                report.supervised_ml ??
                []
              }
              maxRows={100}
            />
          </Section>

          <Section title="Locked-Test Comparison">
            <MetricChart
              rows={(
                report.supervised_ml ??
                []
              ).filter(
                (r) =>
                  r.split ===
                  "locked_test"
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
            <DataTable
              rows={
                report.weak_label_model ??
                []
              }
            />
          </Section>

          <Section title="Weak ML Transfer">
            <DataTable
              rows={
                report.weak_ml_transfer ??
                []
              }
              maxRows={100}
            />
          </Section>

          <Section title="Weak ML Locked-Test Comparison">
            <MetricChart
              rows={(
                report.weak_ml_transfer ??
                []
              )
                .filter(
                  (r) =>
                    r.split ===
                      "locked_test" &&
                    r.granularity ===
                      "fine"
                )
                .sort(
                  (a, b) =>
                    Number(
                      b.macro_f1 ?? 0
                    ) -
                    Number(
                      a.macro_f1 ?? 0
                    )
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
          <DataTable
            rows={
              report.severity_records ??
              []
            }
            maxRows={100}
          />
        </Section>
      )}

      {activeTab === "ranking" && (
        <>
          <Section title="Transformer Ranking">
            <DataTable
              rows={
                report.transformer_ranking ??
                []
              }
              maxRows={100}
            />
          </Section>

          <Section title="Top Transformer Severity">
            <MetricChart
              rows={(
                report.transformer_ranking ??
                []
              )
                .slice(0, 20)
                .map((r) => ({
                  transformer_id:
                    r.transformer_id,
                  severity:
                    Number(
                      r.transformer_overall_severity_level ??
                        0
                    ),
                }))}
              xKey="transformer_id"
              yKey="severity"
              title="Top 20 transformers — current IEEE status"
            />
          </Section>
        </>
      )}

      {activeTab === "stability" && (
        <Section title="History / Stability Evidence">
          <p className="mb-4 text-sm text-teal-600">
            This table reports available historical
            evidence such as number of records,
            history span, worsening transitions,
            recurrence and history sufficiency. It
            is not presented as a fabricated statistical
            ranking-stability score.
          </p>

          <DataTable
            rows={
              report.ranking_stability ??
              []
            }
            maxRows={100}
          />
        </Section>
      )}
    </div>
  );
}