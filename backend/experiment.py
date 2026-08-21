# experiment.py
from __future__ import annotations

import csv
import math
from pathlib import Path
import pandas as pd
from artifact_tool import SpreadsheetFile, Workbook

REPORT_SHEETS = [
    ("Traditional_Individual", "traditional_individual_benchmark.csv"),
    ("Traditional_Combinations", "traditional_combinations_benchmark.csv"),
    ("Traditional_PPM", "traditional_ppm_coverage.csv"),
    ("Traditional_Class_Coverage", "traditional_fault_class_coverage.csv"),
    ("Traditional_Pairwise", "traditional_pairwise_agreement.csv"),
    ("Method_Summary", "traditional_method_summary.csv"),
    ("Supervised_Reference", "supervised_fault_benchmark.csv"),
    ("Weak_Transfer", "weak_transfer_fault_benchmark.csv"),
    ("Weak_Traditional_Hybrid", "weak_traditional_hybrid_benchmark.csv"),
    ("Split_Manifest", "benchmark_split_manifest.csv"),
    ("Label_Conflicts", "external_benchmark_label_conflicts.csv"),
]


def read_csv_file(path):
    path = Path(path)
    if not path.exists(): return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.reader(handle))


def _col_name(number):
    output = ""
    while number:
        number, remainder = divmod(number - 1, 26); output = chr(65 + remainder) + output
    return output


def write_table(sheet, rows, start_row=1, start_col=1, max_width=36):
    if not rows: return
    ncol = max(len(row) for row in rows); normalized = [list(row) + [""] * (ncol - len(row)) for row in rows]
    end_row = start_row + len(normalized) - 1; end_col = start_col + ncol - 1
    ref = f"{_col_name(start_col)}{start_row}:{_col_name(end_col)}{end_row}"; sheet.get_range(ref).values = normalized
    header_ref = f"{_col_name(start_col)}{start_row}:{_col_name(end_col)}{start_row}"; sheet.get_range(header_ref).format.wrap_text = True
    body_start = start_row + 1
    if body_start <= end_row: sheet.get_range(f"{_col_name(start_col)}{body_start}:{_col_name(end_col)}{end_row}").format.wrap_text = True
    sheet.freeze_panes.freeze_rows(1)
    for column in range(start_col, end_col + 1): sheet.get_range(f"{_col_name(column)}{start_row}:{_col_name(column)}{end_row}").format.column_width = max_width


def _add_chart(sheet, source_ref, title, position, chart_type="bar"):
    chart = sheet.charts.add(chart_type, sheet.get_range(source_ref), title=title); chart.title_text = title; chart.set_position(*position); return chart


def _float(value, default=None):
    try:
        x = float(value); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default


def _lookup(rows):
    if not rows: return [], {}
    header = rows[0]; return rows[1:], {name: i for i, name in enumerate(header)}


def _build_summary_sheet(wb, report_dir, processed_dir):
    sheet = wb.worksheets.add("Dashboard"); sheet.get_range("A1:H1").merge(); sheet.get_range("A1").values = [["DGA RESEARCH REPORT — UNLABELED OPERATIONAL DATA"]]
    rows = [
        ["Item", "Definition"],
        ["Operational dataset", "4561-row unlabeled operational DGA dataset; traditional diagnostics are weak labeling functions and student models are trained only on operational weak labels."],
        ["External benchmark", "IEC TC10 (121) + DGA dataset (201), evaluated as an independent labeled benchmark."],
        ["Split protocol", "Train / Development / Locked Test. Development selects methods/models; locked test is never used for selection."],
        ["Primary metric", "Macro F1; accuracy, balanced accuracy, macro precision/recall, weighted F1, coverage and abstain-aware accuracy are also reported."],
        ["Label harmonization", "Spark→D1, Arc→D2, Low/Middle-temperature→T1_T2. T1_T2 is excluded from strict fine scoring but accepted as either T1 or T2 in ambiguity-tolerant scoring."],
        ["Weak supervision", "Snorkel LabelModel when available; EM fallback otherwise. No ground-truth labels from the external benchmark are used for operational weak-label training."],
        ["Student transfer", "Models trained on operational weak labels are applied unchanged to the external labeled benchmark."],
        ["Hybrid evaluation", "Agreement-only hybrid keeps a prediction only when student and unweighted traditional consensus agree exactly; disagreement becomes ABSTAIN. No numeric fusion weight."],
        ["Severity", "IEEE C57.104-2019 rule-derived Status 1/2/3. No invented Status 4 and no arbitrary weighted severity sum."],
        ["Fleet ranking", "One transformer row per asset. Current IEEE status is the primary condition level; standardized current evidence and history are lexicographic tie-break evidence, not weighted into a health score."],
    ]
    write_table(sheet, rows, start_row=3, max_width=100)


def _build_protocol_sheet(wb, benchmark_dir):
    sheet = wb.worksheets.add("Evaluation_Protocol")
    rows = [
        ["Item", "Protocol"],
        ["Benchmark sources", "IEC_TC10_121.csv + DGA dataset.csv"],
        ["Target taxonomy", "NORMAL, PD, D1, D2, T1, T2, T3"],
        ["Ambiguous truth", "T1_T2 accepted as T1 or T2 only in ambiguity-tolerant analysis; strict fine metric excludes it."],
        ["Grouping", "Exact common five-gas signature prevents duplicated identical gas vectors crossing splits."],
        ["Model selection", "Development split only."],
        ["Final estimate", "Locked test only after development selection."],
        ["Operational training", "Unlabeled operational data only; external labels are not used to fit weak students."],
        ["Traditional benchmark", "Every individual LF plus all non-empty 1..7 method combinations."],
        ["ML benchmark", "Gas-only and gas+traditional feature modes across available ML/DL models."],
        ["Hybrid benchmark", "Student/traditional exact agreement gate; no hand-tuned numeric weight."],
        ["PPM coverage", "Empirical observed ppm range in labeled benchmark, not a claimed physical operating range."],
        ["Class coverage", "LF activation rate within each labeled fault class."],
    ]
    write_table(sheet, rows, start_row=2, max_width=96)


def _build_traditional_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_individual_benchmark.csv"); data, idx = _lookup(rows)
    if not data or not {"method", "granularity", "split", "macro_f1", "coverage"}.issubset(idx): return
    selected = [r for r in data if r[idx["granularity"]] == "fine" and r[idx["split"]] == "locked_test" and str(r[idx.get("selected_on_development", -1)]).lower() == "true"]
    if not selected: selected = [r for r in data if r[idx["granularity"]] == "fine" and r[idx["split"]] == "locked_test"]
    sheet = wb.worksheets.add("Chart_Traditional")
    table = [["Method", "Macro F1", "Balanced Accuracy", "Coverage %", "Abstain-aware Accuracy"]]
    for r in selected: table.append([r[idx["method"]], _float(r[idx["macro_f1"]]), _float(r[idx["balanced_accuracy"]]), (_float(r[idx["coverage"]], 0.0) or 0.0) * 100.0, _float(r[idx["overall_accuracy_with_abstain_error"]])])
    write_table(sheet, table, max_width=24); _add_chart(sheet, f"A1:D{len(table)}", "Traditional methods on locked test", ("G2", "P24"), "bar")


def _build_combination_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_combinations_benchmark.csv"); data, idx = _lookup(rows)
    if not data or not {"methods", "granularity", "split", "macro_f1"}.issubset(idx): return
    selected = [r for r in data if r[idx["granularity"]] == "fine" and r[idx["split"]] == "locked_test" and str(r[idx.get("selected_on_development", -1)]).lower() == "true"]
    if not selected: selected = sorted([r for r in data if r[idx["granularity"]] == "fine" and r[idx["split"]] == "locked_test"], key=lambda r: _float(r[idx["macro_f1"]], -1), reverse=True)[:10]
    sheet = wb.worksheets.add("Chart_Combinations")
    table = [["Combination", "Method Count", "Macro F1", "Coverage %", "Abstain-aware Accuracy"]]
    for r in selected: table.append([r[idx["methods"]], int(float(r[idx["method_count"]])), _float(r[idx["macro_f1"]]), (_float(r[idx["coverage"]], 0.0) or 0.0) * 100.0, _float(r[idx["overall_accuracy_with_abstain_error"]])])
    write_table(sheet, table, max_width=42); _add_chart(sheet, f"A1:D{len(table)}", "Best traditional combinations", ("F2", "P28"), "bar")


def _build_model_transfer_chart(wb, benchmark_dir):
    source_files = [("Weak+Student", "weak_transfer_fault_benchmark.csv", "selected_on_dev"), ("Supervised reference", "supervised_fault_benchmark.csv", "selected_on_dev"), ("Hybrid", "weak_traditional_hybrid_benchmark.csv", "selected_on_dev")]
    records = []
    for approach, filename, flag in source_files:
        rows = read_csv_file(benchmark_dir / filename); data, idx = _lookup(rows)
        if not data or "granularity" not in idx or "split" not in idx or "macro_f1" not in idx: continue
        for r in data:
            if r[idx["granularity"]] == "fine" and r[idx["split"]] == "locked_test" and (flag not in idx or str(r[idx[flag]]).lower() == "true"):
                records.append([approach, r[idx.get("model", idx.get("hybrid_policy", 0))], r[idx.get("feature_mode", 0)], _float(r[idx["macro_f1"]]), (_float(r[idx["coverage"]], 0.0) or 0.0) * 100.0, _float(r[idx["overall_accuracy_with_abstain_error"]])])
    if not records: return
    sheet = wb.worksheets.add("Chart_Model_Transfer"); table = [["Approach", "Model/Policy", "Feature Mode", "TEST Macro F1", "Coverage %", "Abstain-aware Accuracy"]] + records
    write_table(sheet, table, max_width=28); _add_chart(sheet, f"A1:E{len(table)}", "ML / weak-transfer / hybrid comparison", ("H2", "Q28"), "bar")


def _build_ppm_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_ppm_coverage.csv"); data, idx = _lookup(rows)
    if not data or not {"method", "gas", "coverage", "min_ppm", "max_ppm"}.issubset(idx): return
    gases = ["h2", "ch4", "c2h6", "c2h4", "c2h2"]; pivot = {}
    for r in data:
        method = r[idx["method"]]; gas = r[idx["gas"]]
        pivot.setdefault(method, {})[gas] = [(_float(r[idx["coverage"]], 0.0) or 0.0) * 100.0, _float(r[idx["min_ppm"]]), _float(r[idx["max_ppm"]])]
    sheet = wb.worksheets.add("Chart_PPM_Coverage"); table = [["Method"] + [f"{g}_Coverage_%" for g in gases]]
    for method in sorted(pivot): table.append([method] + [pivot[method].get(g, [0])[0] for g in gases])
    write_table(sheet, table, max_width=20); _add_chart(sheet, f"A1:F{len(table)}", "Traditional empirical coverage by gas", ("H2", "P24"), "bar")
    range_sheet = wb.worksheets.add("Chart_PPM_Range"); range_table = [["Method", "Gas", "Min ppm", "P05 ppm", "Median ppm", "P95 ppm", "Max ppm", "Observed Range ppm"]]
    for method in sorted(pivot):
        for gas in gases:
            vals = pivot[method].get(gas)
            if vals: range_table.append([method, gas, vals[1], None, None, None, vals[2], vals[2] - vals[1] if vals[1] is not None and vals[2] is not None else None])
    write_table(range_sheet, range_table, max_width=22)


def _build_class_coverage_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_fault_class_coverage.csv"); data, idx = _lookup(rows)
    if not data or not {"method", "fault_class", "class_coverage_percent"}.issubset(idx): return
    classes = sorted({r[idx["fault_class"]] for r in data}); methods = sorted({r[idx["method"]] for r in data}); pivot = {(r[idx["method"]], r[idx["fault_class"]]): _float(r[idx["class_coverage_percent"]], 0.0) or 0.0 for r in data}
    sheet = wb.worksheets.add("Chart_Class_Coverage"); table = [["Method"] + classes]
    for method in methods: table.append([method] + [pivot.get((method, cls), 0.0) for cls in classes])
    write_table(sheet, table, max_width=20); _add_chart(sheet, f"A1:{_col_name(len(classes)+1)}{len(table)}", "Traditional LF coverage by fault class (%)", ("A20", "J40"), "bar")


def _build_ranking_chart(wb, report_dir):
    rows = read_csv_file(report_dir / "transformer_ranking.csv"); data, idx = _lookup(rows)
    if not data or not {"transformer_id", "rank", "transformer_overall_severity_level"}.issubset(idx): return
    top = data[:20]; sheet = wb.worksheets.add("Chart_Transformer_Ranking"); table = [["Transformer", "Fleet Rank", "Current IEEE Status"]]
    for r in top: table.append([r[idx["transformer_id"]], int(float(r[idx["rank"]])), int(float(r[idx["transformer_overall_severity_level"]]))])
    write_table(sheet, table, max_width=28); _add_chart(sheet, f"A1:C{len(table)}", "Top 20 transformer fleet priority", ("E2", "M28"), "bar")


def _build_validation_sheets(wb, report_dir, processed_dir):
    severity_file = processed_dir / "dga_unlabeled_processed.parquet"
    severity = wb.worksheets.add("Severity_Validation")
    write_table(severity, [["Metric", "Value / Interpretation"], ["Independent severity ground truth", "NO"], ["Severity classification accuracy", "NOT COMPUTED — no independent severity labels are supplied."], ["What is externally evaluated", "Fault-type prediction only, against labeled benchmark datasets."], ["Operational severity", "IEEE C57.104-2019 rule-derived Status 1/2/3 with explicit evidence fields."], ["Overall transformer score", "Current condition level; history is retained as explicit evidence and tie-break context, not as an arbitrary weighted sum."]], max_width=88)
    methodology = wb.worksheets.add("Methodology")
    write_table(methodology, [["Item", "Definition"], ["Traditional diagnostics", "Key Gas, IEC-style ratio, Rogers, Doernenburg, Duval Triangle and Duval Pentagon outputs are treated as noisy labeling functions/evidence generators."], ["Weak supervision", "Snorkel LabelModel or EM fallback estimates latent labels from the LF matrix without external ground truth."], ["Student training", "Discriminative students are fitted only on operational weak labels."], ["External evaluation", "External labeled benchmark is never used to train operational student models."], ["Fine-label mismatch", "DGA dataset labels are harmonized to the IEC fault taxonomy; T1_T2 is evaluated both strictly and with set-valued ambiguity tolerance."], ["Class imbalance", "Macro F1 and balanced accuracy are reported; model fitting uses class balancing where supported."], ["Transformer imbalance", "Each transformer is aggregated to exactly one fleet row; single-record transformers are retained and marked as such."], ["Temporal information", "Current status plus standard rate-of-change/delta evidence and historical maximum/recurrence/trend fields are retained without arbitrary historical weights."], ["Ranking", "Lexicographic evidence ordering; no hand-assigned severity weights, no fault-criticality severity weight, no invented Status 4."], ["PPM coverage", "Observed empirical benchmark coverage/range only; it is not interpreted as a physical validity domain beyond the benchmark sample."], ["Hybrid", "Student + traditional agreement gate; disagreement abstains, so no numeric fusion weight is introduced."]], max_width=100)
    if severity_file.exists():
        try:
            df = pd.read_parquet(severity_file); fields = ["transformer_id", "sample_day", "ieee_dga_status", "ieee_dga_status_label", "ieee_dga_status_reason", "ieee_max_standardized_exceedance", "ieee_max_status3_standardized_exceedance", "ieee_standard_trigger_count", "ieee_confirmation_required", "ieee_delta_available", "ieee_rate_available", "ieee_rate_span_months", "ieee_table2_exceeding_gases", "ieee_table4_exceeding_gases"]; fields = [x for x in fields if x in df.columns]
            sh = wb.worksheets.add("Severity_Records"); rows = [fields] + [list(r) for r in df[fields].tail(1000).itertuples(index=False, name=None)]; write_table(sh, rows, max_width=34)
        except Exception: pass


def build_excel_report(report_dir, processed_dir, output_path):
    report_dir = Path(report_dir); processed_dir = Path(processed_dir); output_path = Path(output_path); benchmark_dir = report_dir / "benchmark"
    wb = Workbook.create(); _build_summary_sheet(wb, report_dir, processed_dir); _build_protocol_sheet(wb, benchmark_dir)
    for sheet_name, filename in REPORT_SHEETS:
        rows = read_csv_file(benchmark_dir / filename)
        if rows:
            sheet = wb.worksheets.add(sheet_name); write_table(sheet, rows, max_width=42)
    ranking_rows = read_csv_file(report_dir / "transformer_ranking.csv")
    if ranking_rows:
        sheet = wb.worksheets.add("Transformer_Ranking"); write_table(sheet, ranking_rows, max_width=42)
    _build_validation_sheets(wb, report_dir, processed_dir); _build_traditional_chart(wb, benchmark_dir); _build_combination_chart(wb, benchmark_dir); _build_model_transfer_chart(wb, benchmark_dir); _build_ppm_chart(wb, benchmark_dir); _build_class_coverage_chart(wb, benchmark_dir); _build_ranking_chart(wb, report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True); SpreadsheetFile.export_xlsx(wb).save(output_path); return output_path