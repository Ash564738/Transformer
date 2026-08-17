# text2sql_chat.py
from __future__ import annotations
import logging, os, re, sqlite3, time
from pathlib import Path
from typing import Any
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from data_store import DB_PATH, SAMPLE_COLUMNS, TRANSFORMER_COLUMNS
from chat_fastpath import detect_language, fast_path, obvious_out_of_scope, NO_DATA_REPLY, OUT_OF_SCOPE_REPLY
logger = logging.getLogger(__name__)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("DGA_CHAT_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free").strip()
FALLBACK_MODELS = [x.strip() for x in os.environ.get("DGA_CHAT_FALLBACK_MODELS", "openai/gpt-oss-20b:free,openrouter/free").split(",") if x.strip()]
MAX_ROWS_RETURNED = int(os.environ.get("DGA_CHAT_MAX_ROWS", "200"))
MAX_RESULT_CHARS = int(os.environ.get("DGA_CHAT_MAX_RESULT_CHARS", "18000"))
MAX_HISTORY_TURNS = int(os.environ.get("DGA_CHAT_MAX_HISTORY_TURNS", "4"))
MAX_HISTORY_CHARS = int(os.environ.get("DGA_CHAT_MAX_HISTORY_CHARS", "350"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("DGA_CHAT_TIMEOUT", "8"))
LLM_MAX_TOKENS = int(os.environ.get("DGA_CHAT_MAX_TOKENS", "700"))
SQL_TIMEOUT_SECONDS = float(os.environ.get("DGA_CHAT_SQL_TIMEOUT", "3"))
MAX_SQL_RETRIES = int(os.environ.get("DGA_CHAT_SQL_RETRIES", "1"))
HTTP_REFERER = os.environ.get("DGA_CHAT_HTTP_REFERER", "http://localhost:3000")
APP_TITLE = os.environ.get("DGA_CHAT_APP_TITLE", "DGA Monitor")
OUT_OF_SCOPE_REPLY = {
    "vi": "Tôi chỉ hỗ trợ các câu hỏi liên quan đến máy biến áp, DGA và dữ liệu giám sát.",
    "th": "ฉันรองรับเฉพาะคำถามเกี่ยวกับหม้อแปลงไฟฟ้า การวิเคราะห์ DGA และข้อมูลการตรวจสอบ",
    "en": "I only support questions about transformers, DGA and the monitoring dataset.",
}
NO_DATA_REPLY = {
    "vi": "Chưa có dữ liệu DGA được tải vào hệ thống. Vui lòng tải file và chạy dự đoán trước.",
    "th": "ยังไม่มีข้อมูล DGA ในระบบ กรุณาอัปโหลดไฟล์และรันการประมวลผลก่อน",
    "en": "No DGA dataset has been loaded yet. Please upload a file and run prediction first.",
}
UNSAFE_SQL_REPLY = {
    "vi": "Tôi không thể tạo truy vấn dữ liệu chỉ-đọc an toàn.",
    "th": "ไม่สามารถสร้างคำสั่งค้นหาข้อมูลแบบอ่านอย่างเดียวที่ปลอดภัยได้",
    "en": "I couldn't build a safe read-only query.",
}
QUERY_ERROR_REPLY = {
    "vi": "Đã xảy ra lỗi khi truy vấn dữ liệu.",
    "th": "เกิดข้อผิดพลาดขณะค้นหาข้อมูล",
    "en": "There was an error querying the data.",
}
TEXT_COLUMNS = {"transformer_id","sample_day","tested_day","loc","name","ser","codetx","mfg","severity_label","severity_label_text","status","severity","fault_type","trend","recommended_action","consensus_fault","consensus_fault_traditional","mixed_components","diagnostic_votes","diagnostic_coverage","keygas_fault","iec_fault","rogers_fault","doernenburg_fault","duval_triangle_fault","duval_pentagon_fault","duval_pentagon_p1_fault","duval_pentagon_p2_fault","fault_p1","fault_p2","event_type","tdcg_source","ieee_dga_status_label","ieee_dga_status_reason","ieee_norm_section","ieee_norm_age_bucket","ieee_table1_exceeding_gases","ieee_table2_exceeding_gases","ieee_table3_exceeding_gases","ieee_table4_exceeding_gases","student_fault_label"}
INTEGER_COLUMNS = {"rank","critical_history_count","ieee_dga_status","tdcg_complete","has_event","ieee_confirmation_required","ieee_extreme_dga"}
COLUMN_DESCRIPTIONS = {
    "transformer_id": "Transformer identifier.",
    "sample_day": "DGA sample timestamp.",
    "tested_day": "Tested timestamp if available.",
    "loc": "Station/substation.",
    "name": "Transformer name/tag.",
    "ser": "Serial number.",
    "codetx": "Transformer code.",
    "mfg": "Manufacturer.",
    "h2": "Hydrogen concentration, ppm.",
    "ch4": "Methane concentration, ppm.",
    "c2h6": "Ethane concentration, ppm.",
    "c2h4": "Ethylene concentration, ppm.",
    "c2h2": "Acetylene concentration, ppm.",
    "co": "Carbon monoxide concentration, ppm.",
    "co2": "Carbon dioxide concentration, ppm.",
    "tdcg_raw": "Raw TDCG.",
    "tdcg_recalc": "Recalculated TDCG.",
    "tdcg": "Total dissolved combustible gas, ppm.",
    "o2": "Oxygen reading.",
    "n2": "Nitrogen reading.",
    "o2_n2_ratio": "O2/N2 ratio.",
    "transformer_age_years": "Transformer age.",
    "water": "Water/moisture reading.",
    "temp": "Temperature.",
    "ieee_dga_status": "Rule-based IEEE DGA status code.",
    "ieee_dga_status_label": "IEEE DGA status label.",
    "ieee_dga_status_reason": "IEEE status reason.",
    "ieee_confirmation_required": "IEEE confirmation flag.",
    "ieee_extreme_dga": "Extreme DGA flag.",
    "severity_score": "Internal 0-100 application score.",
    "severity_label": "Machine severity label.",
    "severity_label_text": "Human-readable severity label.",
    "severity_gas_score": "Gas severity component.",
    "severity_trend_score": "Trend severity component.",
    "severity_anomaly_score": "Independent anomaly signal.",
    "consensus_fault": "Final combined fault interpretation.",
    "consensus_fault_traditional": "Traditional consensus.",
    "mixed_components": "Mixed fault components.",
    "diagnostic_confidence": "Diagnostic confidence 0-100.",
    "diagnostic_coverage": "Diagnostic-method coverage.",
    "diagnostic_votes": "Individual diagnostic outputs.",
    "keygas_fault": "Key Gas result.",
    "iec_fault": "IEC 60599 result.",
    "rogers_fault": "Rogers Ratio result.",
    "doernenburg_fault": "Doernenburg result.",
    "duval_triangle_fault": "Duval Triangle result.",
    "duval_pentagon_fault": "Duval Pentagon result.",
    "duval_pentagon_p1_fault": "Duval Pentagon P1 result.",
    "duval_pentagon_p2_fault": "Duval Pentagon P2 result.",
    "fault_p1": "Duval Pentagon P1 result.",
    "fault_p2": "Duval Pentagon P2 result.",
    "student_fault_label": "Student-model fault group.",
    "student_fault_confidence": "Student-model confidence.",
    "anomaly_percentile": "Unsupervised anomaly percentile.",
    "rank": "Fleet risk rank; 1 is highest.",
    "latest_sample_day": "Latest sample timestamp.",
    "latest_score": "Latest sample severity score.",
    "status": "Dashboard status: Normal, Watch, High, Critical.",
    "severity": "Legacy severity label.",
    "fault_type": "Fleet-level fault type.",
    "priority_score": "Fleet priority score.",
    "recommended_action": "Recommended maintenance action.",
    "current_severity": "Current severity component.",
    "historical_severity": "Historical severity.",
    "trend_bonus": "Trend contribution.",
    "critical_history_count": "Historical Critical count.",
    "fault_persistence": "Recent high-severity fraction.",
    "days_since_last_critical": "Days since latest Critical.",
    "fleet_priority_score": "Fleet priority score.",
    "fleet_priority_percent": "Fleet priority percentile.",
    "fleet_priority_rank": "Fleet priority rank.",
    "trend_slope": "Fleet severity trend slope.",
}
def _sql_type(column: str) -> str:
    if column in TEXT_COLUMNS:
        return "TEXT"
    if column in INTEGER_COLUMNS:
        return "INTEGER"
    return "REAL"
def _build_schema_description() -> str:
    blocks = []
    for table, columns in (("transformers", TRANSFORMER_COLUMNS), ("samples", SAMPLE_COLUMNS)):
        purpose = "one row per transformer fleet snapshot" if table == "transformers" else "one row per DGA sample; one transformer can have many samples"
        lines = [f"TABLE {table}", f"Purpose: {purpose}", "Columns:"]
        for column in columns:
            lines.append(f"- {column} ({_sql_type(column)}): {COLUMN_DESCRIPTIONS.get(column, 'Backend field.')}")
        blocks.append("\n".join(lines))
    blocks.append("\n".join([
        "SEMANTICS:",
        "- transformers.priority_score = fleet risk ranking.",
        "- transformers.rank = fleet priority rank; 1 is highest.",
        "- samples.sample_day = chronological DGA sample time.",
        "- transformer_id joins transformers and samples.",
        "- status = Normal | Watch | High | Critical.",
        "- severity_score = internal 0-100 score.",
        "- IEEE DGA status is rule-based and independent of anomaly_percentile.",
        "- ABSTAIN means insufficient diagnostic information, not healthy.",
        "- CTEs, JOINs, UNION/UNION ALL, subqueries and window functions are allowed.",
        "- SQLite date(), datetime() and strftime() are allowed.",
        "- For latest row use ROW_NUMBER() or ORDER BY sample_day DESC.",
        "- For previous row use LAG() or adjacent row_number logic.",
        "- For historical ratios use the full historical denominator, not the filtered subset."]))
    return "\n\n".join(blocks)
SCHEMA_DESCRIPTION = _build_schema_description()
ROUTER_SYSTEM_PROMPT = f"""
You are the semantic router and SQLite query generator for a transformer DGA monitoring dashboard.

Supported languages: Vietnamese, Thai, English.

SCOPE:
Transformer fleet data, DGA samples, dissolved gases, Key Gas, IEC 60599,
Rogers Ratio, Doernenburg, Duval Triangle, Duval Pentagon, IEEE DGA,
severity/status, ranking, anomaly interpretation and transformer maintenance.

OUTPUT:
1. Unrelated question -> exactly OUT_OF_SCOPE
2. Pure conceptual DGA question not requiring current database ->
   NO_QUERY: <answer in user's language>
3. Database question -> exactly ONE SQLite read-only query

VALID SQL:
SELECT ...
WITH ... SELECT ...

Allowed:
CTE, JOIN, LEFT JOIN, INNER JOIN, CROSS JOIN, subquery,
UNION, UNION ALL, CASE, aggregates, GROUP BY, HAVING,
ORDER BY, window functions, date/time/string functions.

Forbidden:
INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, REPLACE,
ATTACH, DETACH, PRAGMA, VACUUM, REINDEX, ANALYZE,
BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE.

Never produce multiple statements.
Never produce comments.
Never invent tables.
Never invent columns.
Never invent transformer IDs.
Do not answer a database question with prose.

IMPORTANT QUERY RULES:
- Most critical/highest fleet risk => transformers.priority_score DESC.
- transformer_id joins transformers and samples.
- Time series => sample_day ASC.
- Latest sample => sample_day DESC.
- Previous sample => LAG() or row_number logic.
- Fault frequency => GROUP BY consensus_fault.
- "critical ratio/history ratio" means numerator over ALL samples in the required history.
- If the question asks for several logical stages, use CTEs.
- If the question asks for first/previous/latest state transitions, use window functions.
- If the question asks agreement among diagnostic methods, compare each method column with consensus_fault.
- If the question asks average rate, calculate AVG(rate_column), not the raw latest rate.
- If the question requires 24 months or N years, derive the date range in SQLite.
- ABSTAIN is not equivalent to healthy.
- Do not use SQL for purely conceptual method explanations.

SCHEMA:
{SCHEMA_DESCRIPTION}
""".strip()
def _sanitize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    result = []
    for item in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        result.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})
    return result
def _build_messages(question: str, history: list[dict[str, str]], context: dict | None) -> list[dict[str, str]]:
    messages = []
    if history:
        selected = history[-2:]
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in selected)
        messages.append({"role": "user", "content": "REFERENCE HISTORY ONLY. Use only to resolve references in CURRENT QUESTION.\n" + history_text})
    if isinstance(context, dict):
        useful = {}
        for key in ("selected_transformer_id", "current_transformer_id", "selected_transformer"):
            if key in context:
                useful[key] = context[key]
        if useful:
            messages.append({"role": "user", "content": "CURRENT DASHBOARD CONTEXT:\n" + str(useful)[:1000]})
    messages.append({"role": "user", "content": "CURRENT QUESTION:\n" + question})
    return messages
def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    name = exc.__class__.__name__.lower()
    return "429" in text or "rate limit" in text or "rate-limited" in text or "too many requests" in text or "ratelimit" in name
def _is_invalid_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not a valid model id" in text or "invalid model" in text or "unknown model" in text or "unavailable for free" in text or "model unavailable" in text
def _is_empty_response_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "empty llm response" in str(exc).lower()
def _candidate_models() -> list[str]:
    models = []
    for model in [MODEL, *FALLBACK_MODELS]:
        model = (model or "").strip()
        if not model:
            continue
        if model not in models:
            models.append(model)
    return models[:3]
def _client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not configured.")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.exception("openai package is not installed.")
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)
def _chat(client, system: str, messages: list[dict[str, str]], max_tokens: int = LLM_MAX_TOKENS) -> str:
    models = _candidate_models()
    if not models:
        raise RuntimeError("No LLM model configured.")
    primary = models[0]
    fallback_models = models[1:3]
    started = time.perf_counter()
    try:
        request_body = {"model": primary, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}, *messages], "extra_headers": {"HTTP-Referer": HTTP_REFERER, "X-Title": APP_TITLE}, "extra_body": {"provider": {"sort": "throughput", "allow_fallbacks": True}}}
        if fallback_models:
            request_body["extra_body"]["models"] = fallback_models
        response = client.chat.completions.create(**request_body)
        elapsed = time.perf_counter() - started
        content = (response.choices[0].message.content if response.choices else "")
        content = (content or "").strip()
        actual_model = getattr(response, "model", primary)
        if not content:
            raise ValueError(f"Empty LLM response from {primary}")
        logger.info("LLM call completed in %.2fs requested=%s actual=%s", elapsed, primary, actual_model)
        return content
    except Exception as exc:
        elapsed = time.perf_counter() - started
        if _is_rate_limit_error(exc):
            logger.warning("LLM rate-limited after %.2fs: %s", elapsed, primary)
        elif _is_invalid_model_error(exc):
            logger.warning("LLM model unavailable after %.2fs: %s", elapsed, primary)
        elif _is_empty_response_error(exc):
            logger.warning("LLM returned empty response after %.2fs: %s", elapsed, primary)
        else:
            logger.warning("LLM primary failed after %.2fs: %s | %s", elapsed, primary, str(exc)[:300])
        for fallback_model in fallback_models:
            fallback_started = time.perf_counter()
            try:
                fallback_request = {"model": fallback_model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}, *messages], "extra_headers": {"HTTP-Referer": HTTP_REFERER, "X-Title": APP_TITLE}, "extra_body": {"provider": {"sort": "throughput", "allow_fallbacks": True}}}
                response = client.chat.completions.create(**fallback_request)
                content = (response.choices[0].message.content if response.choices else "")
                content = (content or "").strip()
                elapsed_fallback = time.perf_counter() - fallback_started
                if not content:
                    raise ValueError(f"Empty LLM response from {fallback_model}")
                actual_model = getattr(response, "model", fallback_model)
                logger.info("LLM fallback completed in %.2fs requested=%s actual=%s", elapsed_fallback, fallback_model, actual_model)
                return content
            except Exception as fallback_exc:
                fallback_elapsed = time.perf_counter() - fallback_started
                logger.warning("LLM fallback failed after %.2fs: %s | %s", fallback_elapsed, fallback_model, str(fallback_exc)[:250])
        raise
FORBIDDEN_SQL_WORDS = {"ATTACH","DETACH","PRAGMA","VACUUM","REINDEX","ANALYZE","CREATE","DROP","ALTER","INSERT","UPDATE","DELETE","REPLACE","TRUNCATE","MERGE","GRANT","REVOKE","BEGIN","COMMIT","ROLLBACK","SAVEPOINT","RELEASE"}
def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not (text.startswith("```") and text.endswith("```")):
        return text
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    body = lines[1:-1]
    if body and body[0].strip().lower() in {"sql", "sqlite"}:
        body = body[1:]
    return "\n".join(body).strip()
def _normalize_sql(text: str) -> str:
    sql = _strip_code_fences(text).strip()
    if sql.lower().startswith("sql"):
        candidate = sql[3:].lstrip(" :\n\r\t")
        if candidate.upper().startswith(("SELECT", "WITH")):
            sql = candidate
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql
def _is_safe_select(sql: str) -> bool:
    if not isinstance(sql, str):
        return False
    sql = sql.strip()
    if not sql:
        return False
    upper = sql.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False
    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        return False
    for word in FORBIDDEN_SQL_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", sql, re.IGNORECASE):
            return False
    return True
def _ensure_limit(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if re.search(r"\bLIMIT\s+\d+(?:\s+OFFSET\s+\d+)?\s*$", sql, re.IGNORECASE):
        return sql
    return "SELECT * FROM (\n" + sql + "\n) AS __dga_result " + f"LIMIT {MAX_ROWS_RETURNED}"
def _authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger_name: str | None) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    if action in allowed:
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = str(arg2 or arg1 or "").lower()
            if function_name in {"load_extension", "writefile"}:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY
def _open_readonly_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError("No DGA dataset has been loaded.")
    uri = f"file:{Path(DB_PATH).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=SQL_TIMEOUT_SECONDS)
    conn.set_authorizer(_authorizer)
    deadline = time.perf_counter() + SQL_TIMEOUT_SECONDS
    def progress_handler():
        return 1 if time.perf_counter() >= deadline else 0
    conn.set_progress_handler(progress_handler, 1000)
    return conn
def _validate_sql(sql: str) -> None:
    if not _is_safe_select(sql):
        raise ValueError("Unsafe read-only query.")
    conn = _open_readonly_connection()
    try:
        conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    finally:
        conn.close()
def _run_sql(sql: str) -> tuple[list[str], list[tuple]]:
    sql = _normalize_sql(sql)
    if not _is_safe_select(sql):
        raise ValueError("Unsafe SQL rejected.")
    bounded_sql = _ensure_limit(sql)
    _validate_sql(bounded_sql)
    conn = _open_readonly_connection()
    try:
        cursor = conn.execute(bounded_sql)
        columns = [item[0] for item in (cursor.description or [])]
        rows = cursor.fetchmany(MAX_ROWS_RETURNED)
        return columns, rows
    finally:
        conn.close()
def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}"
    except (TypeError, ValueError):
        return str(value)
def _python_answer(question: str, language: str, columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        if language == "vi":
            return "Không tìm thấy bản ghi phù hợp."
        if language == "th":
            return "ไม่พบบันทึกที่ตรงกับคำถาม"
        return "No matching records were found."
    records = [dict(zip(columns, row)) for row in rows]
    if len(columns) == 1:
        name = columns[0].lower()
        if "count" in name or name in {"n", "total", "count"}:
            value = records[0][columns[0]]
            if language == "vi":
                return f"Kết quả: {value}."
            if language == "th":
                return f"ผลลัพธ์: {value}"
            return f"Result: {value}."
    if len(records) == 1:
        parts = []
        for column in columns:
            value = records[0].get(column)
            if value is not None:
                parts.append(f"{column}={_fmt(value)}")
        if language == "vi":
            return "Kết quả: " + ", ".join(parts) + "."
        if language == "th":
            return "ผลลัพธ์: " + ", ".join(parts)
        return "Result: " + ", ".join(parts) + "."
    date_column = next((column for column in columns if column.lower() in {"sample_day", "latest_sample_day", "transition_time"}), None)
    if date_column:
        if language == "vi":
            lines = [f"Tìm thấy {len(rows)} bản ghi theo thời gian."]
        elif language == "th":
            lines = [f"พบข้อมูลตามเวลา {len(rows)} รายการ"]
        else:
            lines = [f"Found {len(rows)} chronological records."]
        for record in records[:MAX_ROWS_RETURNED]:
            values = []
            for column in columns:
                if column == date_column:
                    continue
                value = record.get(column)
                if value is not None:
                    values.append(f"{column}={_fmt(value)}")
            lines.append("- " + str(record.get(date_column)) + ": " + ", ".join(values[:12]))
        result = "\n".join(lines)
    else:
        if language == "vi":
            lines = [f"Tìm thấy {len(rows)} bản ghi."]
        elif language == "th":
            lines = [f"พบข้อมูล {len(rows)} รายการ"]
        else:
            lines = [f"Found {len(rows)} records."]
        for record in records[:15]:
            values = []
            for column in columns[:12]:
                value = record.get(column)
                if value is not None:
                    values.append(f"{column}={_fmt(value)}")
            lines.append("- " + ", ".join(values))
        if len(records) > 15:
            remaining = len(records) - 15
            if language == "vi":
                lines.append(f"... và {remaining} bản ghi khác.")
            elif language == "th":
                lines.append(f"... และอีก {remaining} รายการ")
            else:
                lines.append(f"... and {remaining} more records.")
        result = "\n".join(lines)
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + "\n[truncated]"
    return result
def _self_correct_prompt(question: str, sql: str, error: str) -> list[dict[str, str]]:
    prompt = f"""
The SQLite query below failed.

CURRENT QUESTION:
{question}

FAILED SQL:
{sql}

SQL ERROR:
{error}

Generate exactly one corrected SQLite query.

Rules:
- SELECT or WITH ... SELECT only.
- CTE/JOIN/window functions allowed.
- No comments.
- No markdown.
- No explanation.
- Use only the supplied schema.
"""
    return [{"role": "user", "content": prompt.strip()}]
def _fallback_answer(language: str) -> str:
    return QUERY_ERROR_REPLY[language]
def answer_question(question: str, context: dict | None = None, history=None) -> str:
    started = time.perf_counter()
    question = (question or "").strip()
    if not question:
        return ""
    language = detect_language(question)
    logger.info("Text2SQL question (lang=%s): %s", language, question[:160])
    if obvious_out_of_scope(question):
        return OUT_OF_SCOPE_REPLY[language]
    fast_started = time.perf_counter()
    try:
        fast_result = fast_path(question, context)
    except Exception:
        logger.exception("Fast-path failed.")
        fast_result = None
    if fast_result is not None:
        logger.info("Fast-path completed in %.3fs", time.perf_counter() - fast_started)
        logger.info("Total chatbot response time %.3fs", time.perf_counter() - started)
        return fast_result
    if not DB_PATH.exists():
        return NO_DATA_REPLY[language]
    client = _client()
    if client is None:
        return _fallback_answer(language)
    history_messages = _sanitize_history(history)
    messages = _build_messages(question, history_messages, context)
    try:
        route = _chat(client, ROUTER_SYSTEM_PROMPT, messages, max_tokens=LLM_MAX_TOKENS)
    except Exception:
        logger.exception("Text2SQL router call failed.")
        return _fallback_answer(language)
    route = (route or "").strip()
    logger.debug("Router response: %s", route[:1000])
    if route.upper() == "OUT_OF_SCOPE":
        return OUT_OF_SCOPE_REPLY[language]
    if route.upper().startswith("NO_QUERY:"):
        answer = route.split(":", 1)[1].strip()
        if answer:
            return answer
        return _fallback_answer(language)
    sql = _normalize_sql(route)
    if not _is_safe_select(sql):
        logger.warning("Unsafe SQL rejected: %s", sql[:1000])
        return UNSAFE_SQL_REPLY[language]
    for attempt in range(MAX_SQL_RETRIES + 1):
        try:
            columns, rows = _run_sql(sql)
            logger.info("SQL succeeded: %d rows x %d columns", len(rows), len(columns))
            answer = _python_answer(question, language, columns, rows)
            logger.info("Total chatbot response time %.3fs", time.perf_counter() - started)
            return answer
        except (sqlite3.Error, ValueError, FileNotFoundError) as exc:
            logger.warning("SQL execution failed attempt %d/%d: %s", attempt + 1, MAX_SQL_RETRIES + 1, exc)
            if attempt >= MAX_SQL_RETRIES:
                break
            try:
                corrected = _chat(client, ROUTER_SYSTEM_PROMPT, _self_correct_prompt(question, sql, str(exc)), max_tokens=550)
                corrected = _normalize_sql(corrected)
                if not _is_safe_select(corrected):
                    logger.warning("Corrected SQL rejected as unsafe.")
                    break
                sql = corrected
            except Exception:
                logger.exception("SQL correction failed.")
                break
    return _fallback_answer(language)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    tests = [
        "Tìm các transformer mà sample gần nhất có status Critical nhưng sample trước đó không Critical. Cho biết thời điểm chuyển trạng thái, fault trước, fault sau, severity trước và severity sau.",
        "Trong nhóm transformer đang Critical, transformer nào có tỷ lệ sample Critical trong toàn bộ lịch sử cao nhất?",
        "Với mỗi consensus_fault, phương pháp truyền thống nào đồng thuận với fault cuối cùng nhiều nhất? Tính tỷ lệ agreement của Key Gas, IEC 60599, Rogers, Doernenburg, Duval Triangle và Duval Pentagon.",
        "Trong toàn bộ dữ liệu, tìm 10 transformer có tốc độ tăng C2H2 trung bình theo năm cao nhất trong 24 tháng gần nhất. Chỉ tính transformer có ít nhất 3 sample trong khoảng thời gian đó.",
        "Trong 24 tháng gần nhất, transformer nào có C2H2 tăng nhanh nhất nhưng hiện chưa Critical?",
        "Tìm transformer có ít nhất 5 sample mà fault thay đổi từ thermal sang discharge ít nhất 2 lần.",
        "Cho mỗi transformer, tìm sample Critical đầu tiên và số ngày từ sample đó đến sample Critical gần nhất.",
        "Trong nhóm Critical, transformer nào có tỷ lệ Critical cao hơn nhưng priority_score thấp hơn mức trung vị của toàn fleet?",
        "Tìm 20 transformer mà sample mới nhất có fault khác sample trước đó và severity tăng ít nhất 20 điểm.",
        "Với mỗi transformer, tính số phương pháp truyền thống đồng thuận với consensus_fault ở sample mới nhất.",
        "So sánh xu hướng H2, C2H2 và TDCG của 103917 và 8R8409T1 theo thời gian.",
        "Which transformer had the largest increase in C2H2 rate between its previous two samples?",
        "เครื่องไหนมี Critical transition ล่าสุดจาก sample ก่อนหน้าที่ไม่ใช่ Critical?",
        "ช่วยหาหม้อแปลงที่มีสัดส่วน Critical สูงสุดในประวัติทั้งหมด",
    ]
    for question in tests:
        print("=" * 100)
        print("QUESTION:")
        print(question)
        print()
        print(answer_question(question))
        print()