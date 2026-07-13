# ranking.py
import pandas as pd
import numpy as np

def compute_transformer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng xếp hạng transformer dựa trên severity tổng hợp."""
    # Lấy record mới nhất của mỗi transformer
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()
    
    # Tính điểm tổng hợp: 70% điểm hiện tại + 30% trung bình quá khứ
    past_avg = df.groupby("transformer_id")["severity_score"].mean().reset_index()
    past_avg.rename(columns={"severity_score": "avg_past_severity"}, inplace=True)
    ranking = latest_df[["transformer_id", "loc", "name", "severity_score"]].merge(past_avg, on="transformer_id", how="left")
    ranking["final_score"] = 0.7 * ranking["severity_score"] + 0.3 * ranking["avg_past_severity"].fillna(ranking["severity_score"])
    ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking)+1)
    return ranking

def main():
    df = pd.read_parquet("dataset/processed/dga_after_consensus.parquet")  # giả sử đã lưu sau consensus
    ranking = compute_transformer_ranking(df)
    ranking.to_csv("dataset/processed/transformer_ranking.csv", index=False)
    print("Ranking saved.")

if __name__ == "__main__":
    main()