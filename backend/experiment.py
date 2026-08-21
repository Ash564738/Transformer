# experiment.py
from __future__ import annotations
import csv
from pathlib import Path
from artifact_tool import SpreadsheetFile, Workbook

REPORT_SHEETS = [
    ("Traditional_Individual", "traditional_individual_benchmark.csv"),
    ("Traditional_Combinations", "traditional_combinations_benchmark.csv"),
    ("Traditional_PPM", "traditional_ppm_coverage.csv"),
    ("Traditional_Pairwise", "traditional_pairwise_agreement.csv"),
    ("Method_Summary", "traditional_method_summary.csv"),
    ("Supervised_Reference", "supervised_fault_benchmark.csv"),
    ("Weak_Transfer", "weak_transfer_fault_benchmark.csv"),
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
        number, remainder = divmod(number - 1, 26)
        output = chr(65 + remainder) + output
    return output

def write_table(sheet, rows, start_row=1, start_col=1, max_width=36):
    if not rows: return
    column_count = max(len(row) for row in rows)
    normalized = [list(row) + [""] * (column_count - len(row)) for row in rows]
    end_row = start_row + len(normalized) - 1
    end_col = start_col + column_count - 1
    ref = f"{_col_name(start_col)}{start_row}:{_col_name(end_col)}{end_row}"
    sheet.get_range(ref).values = normalized
    header_ref = f"{_col_name(start_col)}{start_row}:{_col_name(end_col)}{start_row}"
    header = sheet.get_range(header_ref)
    header.format.wrap_text = True
    body_start = start_row + 1
    if body_start <= end_row:
        body_ref = f"{_col_name(start_col)}{body_start}:{_col_name(end_col)}{end_row}"
        sheet.get_range(body_ref).format.wrap_text = True
    sheet.freeze_panes.freeze_rows(1)
    for column in range(start_col, end_col + 1):
        column_ref = f"{_col_name(column)}{start_row}:{_col_name(column)}{end_row}"
        sheet.get_range(column_ref).format.column_width = max_width

def _add_chart(sheet, source_ref, title, position, chart_type="bar"):
    chart = sheet.charts.add(chart_type, sheet.get_range(source_ref), title=title)
    chart.title_text = title
    chart.set_position(*position)
    return chart

def _build_traditional_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_individual_benchmark.csv")
    if not rows or len(rows) < 2: return
    header = rows[0]
    required = ["method", "granularity", "macro_f1", "coverage"]
    if any(name not in header for name in required): return
    indexes = {name: header.index(name) for name in required}
    sheet = wb.worksheets.add("Chart_Traditional")
    chart_rows = [["Method", "Macro F1", "Coverage"]]
    for row in rows[1:]:
        if len(row) <= max(indexes.values()): continue
        if row[indexes["granularity"]] != "coarse": continue
        chart_rows.append([
            row[indexes["method"]],
            float(row[indexes["macro_f1"]]) if row[indexes["macro_f1"]] else None,
            float(row[indexes["coverage"]]) if row[indexes["coverage"]] else None,
        ])
    write_table(sheet, chart_rows, max_width=24)
    _add_chart(sheet, f"A1:C{len(chart_rows)}", "Traditional methods: Macro F1 and Coverage", ("E2", "M20"), "bar")

def _build_combination_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_combinations_benchmark.csv")
    if not rows or len(rows) < 2: return
    header = rows[0]
    required = ["methods", "granularity", "split", "macro_f1"]
    if any(name not in header for name in required): return
    indexes = {name: header.index(name) for name in required}
    best = {}
    for row in rows[1:]:
        if len(row) <= max(indexes.values()): continue
        if row[indexes["split"]] != "locked_test": continue
        granularity = row[indexes["granularity"]]
        value = float(row[indexes["macro_f1"]]) if row[indexes["macro_f1"]] else float("nan")
        if granularity not in best or value > best[granularity][1]:
            best[granularity] = (row[indexes["methods"]], value)
    if not best: return
    sheet = wb.worksheets.add("Chart_Combinations")
    table = [["Granularity", "Best Combination", "Macro F1"]]
    for granularity, (methods, value) in sorted(best.items()): table.append([granularity, methods, value])
    write_table(sheet, table, max_width=42)
    _add_chart(sheet, f"A1:C{len(table)}", "Best traditional combination on locked test", ("E2", "M18"), "bar")

def _build_ml_chart(wb, benchmark_dir):
    weak = read_csv_file(benchmark_dir / "weak_transfer_fault_benchmark.csv")
    supervised = read_csv_file(benchmark_dir / "supervised_fault_benchmark.csv")
    if (not weak or len(weak) < 2) and (not supervised or len(supervised) < 2): return
    sheet = wb.worksheets.add("Chart_ML_Transfer")
    table = [["Approach", "Feature Mode", "Macro F1"]]
    if weak and len(weak) > 1:
        header = weak[0]
        required = ["granularity", "model", "feature_mode", "split", "macro_f1"]
        if all(item in header for item in required):
            indexes = {item: header.index(item) for item in required}
            for row in weak[1:]:
                if row[indexes["split"]] != "locked_test": continue
                if row[indexes["granularity"]] != "fine": continue
                table.append(["Weak+" + row[indexes["model"]], row[indexes["feature_mode"]], float(row[indexes["macro_f1"]]) if row[indexes["macro_f1"]] else None])
    if supervised and len(supervised) > 1:
        header = supervised[0]
        required = ["model", "feature_mode", "split", "macro_f1"]
        if all(item in header for item in required):
            indexes = {item: header.index(item) for item in required}
            for row in supervised[1:]:
                if row[indexes["split"]] != "locked_test": continue
                table.append(["Supervised+" + row[indexes["model"]], row[indexes["feature_mode"]], float(row[indexes["macro_f1"]]) if row[indexes["macro_f1"]] else None])
    if len(table) <= 1: return
    write_table(sheet, table, max_width=34)
    _add_chart(sheet, f"A1:C{len(table)}", "External locked-test model comparison", ("E2", "M28"), "bar")

def _build_ppm_chart(wb, benchmark_dir):
    rows = read_csv_file(benchmark_dir / "traditional_ppm_coverage.csv")
    if not rows or len(rows) < 2: return
    header = rows[0]
    required = ["method", "gas", "coverage"]
    if any(item not in header for item in required): return
    indexes = {item: header.index(item) for item in required}
    pivot = {}
    for row in rows[1:]:
        if len(row) <= max(indexes.values()): continue
        method = row[indexes["method"]]
        gas = row[indexes["gas"]]
        coverage = float(row[indexes["coverage"]]) if row[indexes["coverage"]] else 0.0
        pivot.setdefault(method, {})[gas] = coverage
    gases = ["h2", "ch4", "c2h6", "c2h4", "c2h2"]
    sheet = wb.worksheets.add("Chart_PPM_Coverage")
    table = [["Method"] + gases]
    for method, values in sorted(pivot.items()): table.append([method] + [values.get(gas, 0.0) for gas in gases])
    write_table(sheet, table, max_width=20)
    _add_chart(sheet, f"A1:F{len(table)}", "Traditional diagnostic empirical coverage by gas", ("H2", "P24"), "bar")

def _build_ranking_chart(wb, report_dir):
    rows = read_csv_file(report_dir / "transformer_ranking.csv")
    if not rows or len(rows) < 2: return
    header = rows[0]
    required = ["transformer_id", "rank", "transformer_overall_severity_level", "maintenance_priority"]
    if any(item not in header for item in required): return
    indexes = {item: header.index(item) for item in required}
    sheet = wb.worksheets.add("Chart_Transformer_Ranking")
    table = [["Transformer", "Rank", "IEEE Status", "Priority"]]
    for row in rows[1:21]:
        if len(row) <= max(indexes.values()): continue
        table.append([row[indexes["transformer_id"]], int(float(row[indexes["rank"]])), int(float(row[indexes["transformer_overall_severity_level"]])), row[indexes["maintenance_priority"]]])
    write_table(sheet, table, max_width=28)
    _add_chart(sheet, f"A1:C{len(table)}", "Top 20 transformer ranking", ("F2", "N28"), "bar")

def build_excel_report(report_dir, processed_dir, output_path):
    report_dir = Path(report_dir)
    processed_dir = Path(processed_dir)
    output_path = Path(output_path)
    benchmark_dir = report_dir / "benchmark"
    wb = Workbook.create()
    dashboard = wb.worksheets.add("Dashboard")
    dashboard.get_range("A1:H1").merge()
    dashboard.get_range("A1").values = [["DGA FULL AUTOMATIC PIPELINE RESEARCH REPORT"]]
    dashboard.get_range("A3:B15").values = [
        ["Item", "Definition"],
        ["Pipeline", "Upload -> cleaning -> features -> diagnostics -> weak supervision -> students -> anomaly -> IEEE severity -> ranking -> benchmark -> Excel"],
        ["Operational dataset", "Uploaded unlabeled DGA dataset"],
        ["External benchmark", "IEC_TC10_121 + DGA dataset.csv"],
        ["Traditional diagnostics", "Evidence-generating diagnostic labeling functions"],
        ["Weak supervision", "Snorkel LabelModel or EM fallback"],
        ["Student models", "Models trained on operational weak labels"],
        ["Features", "Gas-only and gas + traditional evidence"],
        ["Evaluation", "External labeled train/development/locked-test benchmark"],
        ["Primary metric", "Macro F1"],
        ["Severity", "IEEE C57.104-2019 rule-derived ordinal status"],
        ["Ranking", "Explicit lexicographic evidence ordering"],
        ["Critical", "Operational Status-3 Pareto extension, not IEEE Status 4"],
    ]
    dashboard.get_range("A3:B3").format.wrap_text = True
    for sheet_name, filename in REPORT_SHEETS:
        rows = read_csv_file(benchmark_dir / filename)
        if rows:
            sheet = wb.worksheets.add(sheet_name)
            write_table(sheet, rows)
    ranking = read_csv_file(report_dir / "transformer_ranking.csv")
    if ranking:
        ranking_sheet = wb.worksheets.add("Transformer_Ranking")
        write_table(ranking_sheet, ranking, max_width=38)
    severity_file = processed_dir / "dga_unlabeled_processed.parquet"
    if severity_file.exists():
        try:
            import pandas as pd
            severity_df = pd.read_parquet(severity_file)
            fields = ["transformer_id", "sample_day", "ieee_dga_status", "ieee_dga_status_label", "ieee_dga_status_reason", "ieee_max_standardized_exceedance", "ieee_max_status3_standardized_exceedance", "ieee_standard_trigger_count"]
            fields = [c for c in fields if c in severity_df.columns]
            rows = [fields]
            for _, row in severity_df[fields].tail(1000).iterrows(): rows.append([row.get(field) for field in fields])
            severity_sheet = wb.worksheets.add("Severity_Records")
            write_table(severity_sheet, rows, max_width=32)
        except Exception: pass
    methodology = wb.worksheets.add("Methodology")
    methodology_rows = [
        ["Item", "Definition"],
        ["Upload", "Single upload request triggers the complete pipeline."],
        ["Traditional diagnostics", "Diagnostic evidence only; not treated as ground truth."],
        ["Weak supervision", "Latent labels inferred from multiple diagnostic labeling functions."],
        ["Student training", "Discriminative models trained only from operational weak labels."],
        ["External evaluation", "Labeled benchmark datasets are used for evaluation."],
        ["Primary metric", "Macro F1."],
        ["Severity", "IEEE C57.104-2019 rule-derived Status 0/1/2/3."],
        ["Ranking", "Lexicographic evidence ordering; no weighted health score."],
        ["Critical", "Operational Status-3 Pareto-front extension, not IEEE Status 4."],
        ["Excel", "Generated automatically after all pipeline stages complete."],
    ]
    write_table(methodology, methodology_rows, max_width=60)
    _build_traditional_chart(wb, benchmark_dir)
    _build_combination_chart(wb, benchmark_dir)
    _build_ml_chart(wb, benchmark_dir)
    _build_ppm_chart(wb, benchmark_dir)
    _build_ranking_chart(wb, report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SpreadsheetFile.export_xlsx(wb).save(output_path)
    return output_path