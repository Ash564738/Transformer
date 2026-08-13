import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import DATASET_DIR, DATABASE_DIR, MODEL_DIR, REPORT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ========== CONFIG ==========

LABELED_CSV_PATH = Path(DATASET_DIR) / "IEC_TC10_121.csv"
UNLABELED_CSV_PATH = Path(DATASET_DIR) / "processed" / "accumulated_clean.csv"
MODEL_OUTPUT_PATH = Path(MODEL_DIR) / "fault_supervised_model.joblib"
PREDICTION_OUTPUT_PATH = Path(DATASET_DIR) / "processed" / "unlabeled_predictions.csv"

GAS_COLS = ['h2', 'ch4', 'c2h6', 'c2h4', 'c2h2', 'co', 'co2']

# ========== 1. ĐỌC DỮ LIỆU LABELED ==========
def load_labeled_data(path):
    """Đọc CSV labeled, chuẩn hóa tên cột, trả về DataFrame có cột label."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Đảm bảo cột nhãn tồn tại
    if 'label' not in df.columns:
        raise ValueError("Không tìm thấy cột 'label' trong file labeled")

    df['label'] = df['label'].astype(str).str.strip().str.upper()
    # Chỉ giữ các cột cần thiết
    df = df[GAS_COLS + ['label']].copy()
    # Chuyển gas sang numeric, điền NaN = 0 (có thể thay bằng median)
    df[GAS_COLS] = df[GAS_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    logger.info(f"Đã đọc {len(df)} mẫu labeled, nhãn: {df['label'].unique()}")
    return df

# ========== 2. HUẤN LUYỆN MODEL ==========
def train_model(df):
    """Train RandomForest với cross-validation, lưu artifacts."""
    X = df[GAS_COLS].values
    y_raw = df['label'].values

    # Mã hóa nhãn thành số
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # In báo cáo chi tiết dùng cross_val_predict
    y_pred = cross_val_predict(clf, X_scaled, y, cv=skf)
    print("\n=== Classification Report (Cross-Validation) ===")
    print(classification_report(y, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

    # Train trên toàn bộ dữ liệu
    clf.fit(X_scaled, y)
    logger.info("Model đã huấn luyện xong trên toàn bộ labeled data.")

    # Lưu artifacts
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': clf,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': GAS_COLS,
    }, MODEL_OUTPUT_PATH)
    logger.info(f"Đã lưu model tại: {MODEL_OUTPUT_PATH}")

    return clf, scaler, le

# ========== 3. ĐỌC DỮ LIỆU UNLABELED ==========
def load_unlabeled_data(path):
    """Đọc file unlabeled (CSV/Excel), trả về DataFrame có đủ gas columns."""
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]

    # Đổi tên nếu cần (các tên cột có thể khác)
    rename_map = {
        'hydrogen': 'h2',
        'methane': 'ch4',
        'ethane': 'c2h6',
        'ethylene': 'c2h4',
        'acetylene': 'c2h2',
        'carbon_monoxide': 'co',
        'carbon_dioxide': 'co2',
    }
    df.rename(columns=rename_map, inplace=True)

    # Kiểm tra các cột gas bắt buộc
    missing = [c for c in GAS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột khí: {missing}. Các cột hiện có: {df.columns.tolist()}")

    # Chuyển numeric, fill NaN = 0 (hoặc median từ train, nhưng tạm dùng 0)
    df[GAS_COLS] = df[GAS_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    logger.info(f"Đã đọc {len(df)} mẫu unlabeled.")
    return df

# ========== 4. DỰ ĐOÁN ==========
def predict_unlabeled(artifacts, df):
    """Dự đoán nhãn cho unlabeled DataFrame."""
    clf = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['label_encoder']
    feature_cols = artifacts['feature_cols']

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    y_pred = clf.predict(X_scaled)
    proba = clf.predict_proba(X_scaled).max(axis=1)

    df['predicted_label'] = le.inverse_transform(y_pred)
    df['confidence'] = proba

    # Thêm xác suất từng lớp (tùy chọn)
    proba_df = pd.DataFrame(
        clf.predict_proba(X_scaled),
        columns=[f'prob_{cls}' for cls in le.classes_]
    )
    df = pd.concat([df, proba_df], axis=1)

    return df

# ========== MAIN ==========
if __name__ == '__main__':
    # 1. Train
    labeled_df = load_labeled_data(LABELED_CSV_PATH)
    clf, scaler, le = train_model(labeled_df)

    # 2. Predict unlabeled (chỉ chạy nếu file tồn tại)
    if UNLABELED_CSV_PATH.exists():
        unlabeled_df = load_unlabeled_data(UNLABELED_CSV_PATH)
        results = predict_unlabeled(
            joblib.load(MODEL_OUTPUT_PATH), unlabeled_df
        )
        results.to_csv(PREDICTION_OUTPUT_PATH, index=False)
        logger.info(f"Đã lưu kết quả dự đoán tại: {PREDICTION_OUTPUT_PATH}")
        print("\n=== 5 mẫu dự đoán đầu tiên ===")
        print(results[['predicted_label', 'confidence'] + GAS_COLS].head())
    else:
        logger.warning(f"File unlabeled không tồn tại: {UNLABELED_CSV_PATH}")