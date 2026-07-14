# ranking.py
import pandas as pd

def build_transformer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    past_avg = df.groupby("transformer_id")["severity_score"].mean().reset_index()
    past_avg.rename(columns={"severity_score": "avg_past_severity"}, inplace=True)

    ranking = latest_df[
    [
    "transformer_id",
    "loc",
    "name",
    "severity_score",
    "severity_label",
    "consensus_fault",
    "sample_day",
    "diagnostic_confidence"
    ]
    ].merge(
        past_avg, on="transformer_id", how="left"
    )
    ranking["final_score"] = (
        0.65 * ranking["severity_score"]
        +
        0.25 * ranking["avg_past_severity"].fillna(
            ranking["severity_score"]
        )
        +
        0.10 * (100-ranking["diagnostic_confidence"])/10
    )
    ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)

    def trend_label(score):
        if score > 12: return "worsening"
        if score > 6: return "moderate"
        return "stable"
    ranking["trend"] = ranking["final_score"].apply(
        trend_label
    )
    def action(row):
        if row["final_score"] > 10:
            return f"Inspect urgently, likely {row['consensus_fault']}"
        elif row["final_score"] > 5:
            return "Increase monitoring frequency"
        else:
            return "Routine monitoring"
    ranking["recommended_action"] = ranking.apply(action, axis=1)
    return ranking