# tasks.py
from celery_worker import celery
from inference_service import process_dataframe
import pandas as pd
import json

@celery.task(name="tasks.predict_async")
def predict_async(file_bytes, filename):
    # Giả sử file_bytes là nội dung file, bạn cần parse tùy loại
    if filename.endswith(".csv"):
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    else:
        df = pd.read_excel(pd.io.common.BytesIO(file_bytes), engine="openpyxl")
    result = process_dataframe(df)
    return result