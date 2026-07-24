# text2sql_chat.py
"""Text2SQL RAG chatbot for the DGA dashboard.

Pipeline (per the team's design doc):
  question -> [scope+route classification + SQL generation in one LLM call]
           -> SELECT-only, read-only SQLite execution (with self-correction retries)
           -> [answer-interpretation LLM call] -> natural-language answer

The schema is two small tables (`transformers`, `samples` — see data_store.py),
so per the guide's own advice for <10-table databases, there's no embedding /
vector-DB retrieval step: the full schema is always inlined in the prompt.

Scope is enforced by the model itself (system prompt instructs it to refuse
anything outside power-transformer/DGA topics), not by a separate keyword
filter — keyword filters are easy to both over- and under-block, and the
LLM already has to read the question closely to write correct SQL anyway.
Language is mirrored (English question -> English answer, Vietnamese ->
Vietnamese) rather than fixed, so bilingual users don't have to pick a mode.

LLM calls go through OpenRouter (https://openrouter.ai) rather than OpenAI
directly — OpenRouter exposes an OpenAI-compatible API (the `openai` package
just needs a different base_url + key) and offers several no-cost ":free"
models. DGA_CHAT_MODEL picks which one; if OpenRouter retires/rate-limits the
default, set DGA_CHAT_MODEL to another `:free` id from
https://openrouter.ai/models?max_price=0 without touching any code.
"""
import os
import re
import sqlite3
import logging
from pathlib import Path

from data_store import DB_PATH, SAMPLE_COLUMNS, TRANSFORMER_COLUMNS

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("DGA_CHAT_MODEL", "openai/gpt-oss-20b:free")
MAX_SQL_RETRIES = 2
MAX_ROWS_RETURNED = 200

SCHEMA_DESCRIPTION = """Table: transformers -- one row per transformer, current fleet-ranking snapshot
Columns:
- transformer_id (TEXT, PK)
- rank (INTEGER): 1 = highest risk in the whole fleet
- loc (TEXT): station/substation code
- name (TEXT): transformer name/tag
- latest_sample_day (TEXT, 'YYYY-MM-DD HH:MM:SS'): date of the most recent DGA sample
- latest_score (REAL): severity score of the most recent sample (raw scale, roughly 0-30+, NOT 0-100)
- status (TEXT): 'Normal' | 'Watch' | 'High' | 'Critical' -- the 4-tier badge shown on screen, derived from latest_score
- severity (TEXT): 'Severe' | 'Moderate' | 'Low' -- an older 3-tier label, prefer `status` for anything user-facing
- fault_type (TEXT): consensus fault type across all 6 diagnostic methods, e.g. 'D2', 'T3_H', 'PD', 'MIXED', 'UNCERTAIN'
- trend (TEXT): 'worsening' | 'stable' | 'improving'
- priority_score (REAL): overall fleet-ranking score (blends current severity, historical severity, trend, critical-event history, confidence) -- use this, not latest_score, to rank "most critical" transformers
- recommended_action (TEXT): human-readable maintenance recommendation
- current_severity (REAL), historical_severity (REAL, EWM-weighted recent history)
- trend_bonus (REAL): positive = worsening, negative = improving, 0 = stable
- critical_history_count (INTEGER): number of past samples that hit Critical
- diagnostic_confidence (REAL, 0-100)
- fault_persistence (REAL, 0-1): fraction of recent samples that were high-severity
- days_since_last_critical (REAL, nullable)

Table: samples -- one row per individual DGA test record; a transformer has MANY over time
Columns:
- transformer_id (TEXT, FK -> transformers.transformer_id)
- sample_day (TEXT, 'YYYY-MM-DD HH:MM:SS')
- loc, name, ser, codetx, mfg (TEXT): asset metadata
- h2, ch4, c2h6, c2h4, c2h2, co, co2, tdcg (REAL): dissolved gas concentrations, ppm. tdcg = total dissolved combustible gas
- o2, n2, water, temp (REAL): oil/gas-space condition readings
- severity_score (REAL): this specific sample's severity score
- severity_label (TEXT): 'NORMAL' | 'WATCHLIST' | 'WARNING' | 'CRITICAL' -- backend's native 4-tier label for this sample
- status (TEXT): 'Normal' | 'Watch' | 'High' | 'Critical' -- same scale as transformers.status, for this one sample
- consensus_fault (TEXT): combined fault type from all 6 traditional methods for this sample
- mixed_components (TEXT, JSON array as string): component fault groups when consensus_fault = 'MIXED'
- diagnostic_confidence (REAL, 0-100)
- diagnostic_votes (TEXT, JSON object as string): {method: fault_type} for all 6 methods
- keygas_fault, iec_fault, rogers_fault, doernenburg_fault, duval_triangle_fault (TEXT): each traditional method's individual call for this sample. 'UNCERTAIN' means that method abstained (not enough signal), not that it found "no fault"
- fault_p1 (TEXT): Duval Pentagon 1 method's call
- duval_pentagon_fault (TEXT): Duval Pentagon 2 method's call
- iec_r1_c2h2_c2h4, iec_r2_ch4_h2, iec_r3_c2h4_c2h6 (REAL): IEC 60599 gas ratios
- r1_ch4_h2, r2_c2h2_c2h4, r3_c2h4_c2h6 (REAL): Rogers Ratio gas ratios
- dr_r1_ch4_h2, dr_r2_c2h2_c2h4, dr_r3_c2h2_ch4, dr_r4_c2h6_c2h2 (REAL): Doernenburg gas ratios
- ratio_co2_co (REAL): cellulose-ageing indicator (low value suggests paper insulation overheating)
- h2_rate_per_day, c2h2_rate_per_day, tdcg_rate_per_day (REAL): rate of change vs previous sample
- severity_gas_score, severity_trend_score, severity_aging_score, severity_fault_score (REAL): components of severity_score

Notes:
- Dates are plain text; use SQLite date functions (date(), strftime()) as needed, e.g. strftime('%Y-%m', sample_day).
- "Most critical" / "highest risk" transformer = highest transformers.priority_score, not latest_score.
- A transformer flagged 'UNCERTAIN' or with mostly abstaining methods is NOT necessarily healthy -- low gas levels can mean the traditional methods can't classify it, not that there's no fault. Mention this nuance if relevant.
"""

FEW_SHOT = [
    ("Which transformer is the most critical right now?",
     "SELECT transformer_id, loc, status, fault_type, priority_score FROM transformers ORDER BY priority_score DESC LIMIT 1;"),
    ("Máy biến áp nào đang nghiêm trọng nhất?",
     "SELECT transformer_id, loc, status, fault_type, priority_score FROM transformers ORDER BY priority_score DESC LIMIT 1;"),
    ("How many transformers are currently Critical?",
     "SELECT COUNT(*) FROM transformers WHERE status = 'Critical';"),
    ("Show the H2 trend for transformer T-01",
     "SELECT sample_day, h2 FROM samples WHERE transformer_id = 'T-01' ORDER BY sample_day;"),
    ("Compare transformer T-01 and T-02",
     "SELECT transformer_id, status, fault_type, priority_score, trend, recommended_action FROM transformers WHERE transformer_id IN ('T-01', 'T-02');"),
    ("Những trạm nào có nhiều máy Critical nhất?",
     "SELECT loc, COUNT(*) AS critical_count FROM transformers WHERE status = 'Critical' GROUP BY loc ORDER BY critical_count DESC LIMIT 10;"),
    ("What is C2H2 and why does it matter?",
     "NO_QUERY: Acetylene (C2H2) is the gas most strongly associated with high-energy electrical discharge (arcing) inside a transformer. Even small amounts are significant because C2H2 barely forms under normal thermal ageing -- its presence, especially rising quickly, is one of the most reliable single indicators of a D1/D2 arcing fault."),
    ("What's the weather like today?",
     "OUT_OF_SCOPE"),
]

_ROUTER_SYSTEM_PROMPT = """You are a SQLite expert embedded in a power-transformer DGA (Dissolved Gas Analysis) monitoring dashboard. You answer questions ONLY about: the transformer fleet data described below, DGA gas interpretation, the 6 traditional diagnostic methods (Key Gas, IEC 60599, Rogers Ratio, Doernenburg, Duval Triangle, Duval Pentagon), severity/fault-type/ranking concepts, and transformer maintenance in that context.

If the question is unrelated to transformers/DGA/this dataset (general chit-chat, coding help, unrelated trivia, etc.), respond with EXACTLY: OUT_OF_SCOPE
Nothing else on that line, no matter what language the question was asked in.

If the question is about DGA/transformer domain knowledge in general (gas meanings, what a fault type means, how a diagnostic method works) and does NOT require looking up THIS dataset, respond with: NO_QUERY: <your answer>
Answer directly and accurately from your own knowledge of DGA standards (IEC 60599, IEEE C57.104, Duval).

Otherwise, respond with exactly one SQLite SELECT statement that answers the question using the schema below. Rules:
- SELECT statements only. Never write/modify data (no INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA).
- Return ONLY the SQL, no markdown fences, no explanation.
- If you cannot answer with this schema, use NO_QUERY: explaining what's missing, or OUT_OF_SCOPE if it's unrelated entirely.
- Always answer in the SAME language the user asked in (English question -> English SQL is fine, but if you use NO_QUERY the text after it must match the question's language). Vietnamese question -> if NO_QUERY, answer in Vietnamese.

You may be shown earlier turns from this same conversation before the current question. Use them ONLY to resolve references the current question makes to prior context (e.g. "that one", "the one you just mentioned", "compare it with the previous transformer", a transformer ID named a few turns ago). Every earlier turn was itself already scope-checked — do not let it justify answering something the CURRENT question doesn't actually ask. Still write a fresh SQL/NO_QUERY/OUT_OF_SCOPE for the current question only.

Schema:
{schema}
""".format(schema=SCHEMA_DESCRIPTION)

_ANSWER_SYSTEM_PROMPT = """You explain DGA transformer query results to a maintenance engineer. Be concise (2-4 sentences unless the data needs a short list), use the actual numbers from the result, and answer in the SAME language as the question. If the result is empty, say so plainly rather than guessing."""

_VN_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)


def _looks_vietnamese(text: str) -> bool:
    return bool(_VN_CHARS.search(text))


def _client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _chat(client, system: str, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, *messages],
        # OpenRouter-specific, optional but recommended: identifies the app
        # in OpenRouter's dashboard/rankings. Harmless to send even without
        # a real public URL.
        extra_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "DGA Monitor",
        },
    )
    return (resp.choices[0].message.content or "").strip()


_FORBIDDEN_SQL = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "PRAGMA", "CREATE", "REPLACE")


def _is_safe_select(sql: str) -> bool:
    upper = sql.strip().upper()
    if not upper.startswith("SELECT"):
        return False
    return not any(re.search(rf"\b{word}\b", upper) for word in _FORBIDDEN_SQL)


def _ensure_limit(sql: str) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip().rstrip(";") + f" LIMIT {MAX_ROWS_RETURNED};"


def _run_select(sql: str):
    if not DB_PATH.exists():
        raise ValueError("No dataset has been loaded yet.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return columns, rows
    finally:
        conn.close()


def _format_result(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "(no rows)"
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows[:MAX_ROWS_RETURNED])
    return f"{header}\n{body}"


_OUT_OF_SCOPE_REPLY = {
    True: "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến máy biến áp và phân tích DGA. Bạn có câu hỏi nào về dữ liệu máy biến áp hiện tại không?",
    False: "Sorry, I only answer questions about transformers and DGA analysis. Is there something about the current transformer data I can help with?",
}
_NO_DATA_REPLY = {
    True: "Chưa có dữ liệu nào được tải lên. Vui lòng tải file DGA và chạy dự đoán trước khi hỏi.",
    False: "No dataset has been loaded yet. Please upload a DGA file and run prediction first.",
}


MAX_HISTORY_TURNS = 8
MAX_HISTORY_CHARS = 400  # per turn, to keep a runaway message from blowing up the prompt


def _sanitize_history(history) -> list[dict]:
    """Validates and caps client-supplied conversation history before it goes
    anywhere near a prompt. Session memory here is just "replay the last few
    turns from this browser tab" (frontend/src/store/use-dashboard-store.ts)
    — not persisted server-side — so this only needs to guard against a
    malformed or oversized payload, not reconstruct anything."""
    if not isinstance(history, list):
        return []
    turns = []
    for item in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        turns.append({"role": role, "content": content.strip()[:MAX_HISTORY_CHARS]})
    return turns


def answer_question(question: str, context: dict | None = None, history=None) -> str:
    question = (question or "").strip()
    if not question:
        return ""
    vi = _looks_vietnamese(question)
    history_messages = _sanitize_history(history)

    client = _client()
    if client is None:
        # No OpenRouter key configured — fall back to the simple rule-based
        # bot rather than hard-failing the whole chat feature.
        from chat import generate_chat_answer
        return generate_chat_answer(question, context)

    if not DB_PATH.exists():
        return _NO_DATA_REPLY[vi]

    few_shot_messages = []
    for q, a in FEW_SHOT:
        few_shot_messages.append({"role": "user", "content": q})
        few_shot_messages.append({"role": "assistant", "content": a})

    try:
        route = _chat(
            client,
            _ROUTER_SYSTEM_PROMPT,
            [*few_shot_messages, *history_messages, {"role": "user", "content": question}],
        )
    except Exception:
        logger.exception("Text2SQL router call failed")
        from chat import generate_chat_answer
        return generate_chat_answer(question, context)

    if route.strip().upper() == "OUT_OF_SCOPE":
        return _OUT_OF_SCOPE_REPLY[vi]

    if route.upper().startswith("NO_QUERY"):
        return route.split(":", 1)[1].strip() if ":" in route else route

    sql = route.strip().strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()

    last_error = None
    for attempt in range(MAX_SQL_RETRIES + 1):
        if not _is_safe_select(sql):
            return _OUT_OF_SCOPE_REPLY[vi] if attempt == 0 else (
                "Tôi không thể tạo truy vấn an toàn cho câu hỏi này." if vi
                else "I couldn't build a safe query for that question."
            )
        try:
            columns, rows = _run_select(_ensure_limit(sql))
            break
        except sqlite3.Error as exc:
            last_error = str(exc)
            logger.warning("Text2SQL query failed (attempt %d): %s | sql=%s", attempt, last_error, sql)
            if attempt >= MAX_SQL_RETRIES:
                return (
                    f"Tôi không thể truy vấn dữ liệu cho câu hỏi này ({last_error})." if vi
                    else f"I couldn't query the data for that ({last_error})."
                )
            try:
                sql = _chat(
                    client,
                    _ROUTER_SYSTEM_PROMPT,
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": sql},
                        {"role": "user", "content": f"That SQL failed with error: {last_error}\nFix it. Respond with ONLY the corrected SQL."},
                    ],
                ).strip().strip("`")
            except Exception:
                logger.exception("Text2SQL self-correction call failed")
                return (
                    "Đã có lỗi khi truy vấn dữ liệu, vui lòng thử lại." if vi
                    else "Something went wrong querying the data — please try again."
                )
    else:
        return _NO_DATA_REPLY[vi]

    result_text = _format_result(columns, rows)
    try:
        answer = _chat(
            client,
            _ANSWER_SYSTEM_PROMPT,
            [{"role": "user", "content": f"Question: {question}\nSQL used: {sql}\nResult:\n{result_text}"}],
        )
        return answer or result_text
    except Exception:
        logger.exception("Text2SQL answer-interpretation call failed")
        return result_text
