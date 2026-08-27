# chat_fastpath.py
from __future__ import annotations
import re, sqlite3
from pathlib import Path
from typing import Any
from data_store import DB_PATH

# ============================================================
# LANGUAGE
# ============================================================
_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
    r"ỳýỵỷỹđ]",
    re.IGNORECASE,
)
_TH_CHARS = re.compile(r"[\u0E00-\u0E7F]")
def detect_language(text: str) -> str:
    text = text or ""
    if _TH_CHARS.search(text): return "th"
    if _VI_CHARS.search(text): return "vi"
    return "en"

# ============================================================
# REPLIES
# ============================================================
NO_DATA_REPLY = {
    "vi": "Chưa có dữ liệu DGA trong hệ thống. Vui lòng tải file và chạy dự đoán trước.",
    "th": "ยังไม่มีข้อมูล DGA ในระบบ กรุณาอัปโหลดไฟล์และรันการประมวลผลก่อน",
    "en": "No DGA dataset has been loaded yet. Please upload a file and run prediction first.",
}
OUT_OF_SCOPE_REPLY = {
    "vi": "Tôi chỉ hỗ trợ các câu hỏi liên quan đến máy biến áp, DGA và dữ liệu giám sát.",
    "th": "ฉันรองรับเฉพาะคำถามเกี่ยวกับหม้อแปลงไฟฟ้า การวิเคราะห์ DGA และข้อมูลการตรวจสอบ",
    "en": "I only support questions about transformers, DGA and the monitoring dataset.",
}

# ============================================================
# DGA METHODS
# ============================================================
ALL_DGA_METHODS = {
    "Key Gas": "keygas_fault",
    "IEC 60599": "iec_fault",
    "Rogers Ratio": "rogers_fault",
    "Doernenburg": "doernenburg_fault",
    "Duval Triangle": "duval_triangle_fault",
    "Duval Pentagon": "duval_pentagon_fault",
}
DGA_METHOD_ALIASES = {
    "key gas": "Key Gas", "keygas": "Key Gas", "key gas method": "Key Gas",
    "iec 60599": "IEC 60599", "iec60599": "IEC 60599", "iec": "IEC 60599",
    "rogers ratio": "Rogers Ratio", "rogers": "Rogers Ratio",
    "doernenburg": "Doernenburg", "dörnenburg": "Doernenburg", "dorn": "Doernenburg",
    "duval triangle 1": "Duval Triangle", "duval triangle": "Duval Triangle", "triangle": "Duval Triangle",
    "duval pentagon": "Duval Pentagon", "pentagon": "Duval Pentagon",
}
def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(p.lower() in lowered for p in phrases)
def extract_requested_dga_methods(question: str) -> list[tuple[str, str]]:
    text = (question or "").lower()
    found = []
    seen = set()
    for alias, method_name in sorted(DGA_METHOD_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in text and method_name not in seen:
            seen.add(method_name)
            found.append((method_name, ALL_DGA_METHODS[method_name]))
    all_terms = [
        "all methods", "all dga methods", "all diagnostic methods",
        "all traditional methods", "all six methods", "all 6 methods",
        "tất cả phương pháp", "tất cả các phương pháp", "toàn bộ phương pháp",
        "đủ 6 phương pháp", "6 phương pháp", "ทุกวิธี", "ทุกวิธี dga",
        "ทุกวิธีการวิเคราะห์", "ทั้ง 6 วิธี",
    ]
    if _contains_any(text, all_terms): return list(ALL_DGA_METHODS.items())
    return found

# ============================================================
# TRANSFORMER IDs
# ============================================================
TRANSFORMER_ID_PATTERNS = [
    re.compile(r"\b(?:máy|transformer|tx|tr)\s*#?\s*([A-Za-z0-9_-]+)", re.IGNORECASE),
    re.compile(r"\b(?:transformer_id|transformer|máy)\s*[=:]\s*['\"]?([A-Za-z0-9_-]+)", re.IGNORECASE),
    re.compile(r"\b(?:เครื่อง|หม้อแปลง)\s*#?\s*([A-Za-z0-9_-]+)", re.IGNORECASE),
]
def extract_transformer_ids(question: str) -> list[str]:
    text = question or ""
    ids = []
    for pattern in TRANSFORMER_ID_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).strip()
            if value not in ids: ids.append(value)
    for match in re.finditer(r"\b\d{5,12}\b", text):
        value = match.group(0)
        if value not in ids: ids.append(value)
    return ids
def extract_transformer_id(question: str) -> str | None:
    ids = extract_transformer_ids(question)
    return ids[0] if ids else None

# ============================================================
# DGA KNOWLEDGE
# ============================================================
METHOD_KNOWLEDGE = {
    "Key Gas": {
        "vi": "Key Gas là phương pháp định tính dựa trên các khí đặc trưng để nhận diện nhóm sự cố. H2 thường liên quan đến discharge, CH4/C2H6 hỗ trợ nhận diện thermal activity, C2H4 nổi bật hơn ở thermal fault nhiệt độ cao, C2H2 đặc biệt quan trọng với arcing, còn CO/CO2 liên quan nhiều hơn đến lão hóa cellulose.",
        "th": "Key Gas เป็นวิธีเชิงคุณภาพที่ใช้รูปแบบก๊าซสำคัญเพื่อบ่งชี้กลุ่ม fault โดย H2 มักสัมพันธ์กับ discharge, CH4/C2H6 กับ thermal activity, C2H4 กับ thermal fault ที่รุนแรงขึ้น และ C2H2 กับ arcing; CO/CO2 เกี่ยวข้องกับการเสื่อมของ cellulose",
        "en": "Key Gas is a qualitative method using characteristic gas patterns. H2 is commonly associated with discharge, CH4/C2H6 with thermal activity, C2H4 with higher-temperature thermal faults, C2H2 with arcing, and CO/CO2 with cellulose insulation ageing.",
    },
    "IEC 60599": {
        "vi": "IEC 60599 sử dụng tổ hợp các tỷ số khí như C2H2/C2H4, CH4/H2 và C2H4/C2H6 để phân biệt discharge và thermal conditions. Phương pháp dựa trên mẫu hình nhiều khí và có thể không kết luận khi dữ liệu không đủ.",
        "th": "IEC 60599 ใช้อัตราส่วนก๊าซ เช่น C2H2/C2H4, CH4/H2 และ C2H4/C2H6 เพื่อแยก discharge และ thermal conditions โดยพิจารณารูปแบบหลายก๊าซและอาจไม่สรุปผลเมื่อข้อมูลไม่เพียงพอ",
        "en": "IEC 60599 uses gas ratios such as C2H2/C2H4, CH4/H2 and C2H4/C2H6 to distinguish discharge and thermal conditions. It relies on multi-gas ratio patterns and may not classify insufficient data.",
    },
    "Rogers Ratio": {
        "vi": "Rogers Ratio sử dụng các tỷ số CH4/H2, C2H2/C2H4 và C2H4/C2H6 để phân vùng fault. Ưu điểm là đơn giản và trực tiếp, nhưng tỷ số có thể nhạy khi mẫu số rất nhỏ.",
        "th": "Rogers Ratio ใช้ CH4/H2, C2H2/C2H4 และ C2H4/C2H6 เพื่อจำแนก fault มีข้อดีคือเรียบง่าย แต่ ratio อาจไวมากเมื่อค่าตัวส่วนต่ำ",
        "en": "Rogers Ratio classifies faults using CH4/H2, C2H2/C2H4 and C2H4/C2H6. It is simple and direct, but ratios can become sensitive when denominator concentrations are very small.",
    },
    "Doernenburg": {
        "vi": "Doernenburg sử dụng tỷ số của các khí cháy để phân biệt thermal và electrical faults. Phương pháp có điều kiện áp dụng chặt hơn và có thể ABSTAIN khi dữ liệu không đủ.",
        "th": "Doernenburg ใช้อัตราส่วนของก๊าซติดไฟหลายตัวเพื่อแยก thermal และ electrical fault และมีเงื่อนไขการใช้งานที่ค่อนข้างเข้ม จึงอาจ ABSTAIN เมื่อข้อมูลไม่เพียงพอ",
        "en": "Doernenburg uses combustible-gas ratios to distinguish thermal and electrical faults. Its applicability conditions are stricter, so it may abstain when the gas pattern is insufficient.",
    },
    "Duval Triangle": {
        "vi": "Duval Triangle 1 sử dụng tỷ lệ CH4, C2H4 và C2H2 trên một ternary diagram. Các vùng fault chính gồm PD, D1, D2, T1, T2 và T3.",
        "th": "Duval Triangle 1 ใช้สัดส่วน CH4, C2H4 และ C2H2 บน ternary diagram โดยมีโซนหลัก PD, D1, D2, T1, T2 และ T3",
        "en": "Duval Triangle 1 uses CH4, C2H4 and C2H2 on a ternary diagram. Its main fault zones are PD, D1, D2, T1, T2 and T3.",
    },
    "Duval Pentagon": {
        "vi": "Duval Pentagon sử dụng H2, CH4, C2H6, C2H4 và C2H2 nên giữ lại nhiều thông tin thành phần khí hơn Duval Triangle. Trong backend này, duval_pentagon_fault là kết quả tổng hợp; P1/P2 được lưu riêng.",
        "th": "Duval Pentagon ใช้ H2, CH4, C2H6, C2H4 และ C2H2 จึงใช้ข้อมูลก๊าซมากกว่า Duval Triangle ใน backend นี้ duval_pentagon_fault เป็นผลรวม และ P1/P2 ถูกเก็บแยก",
        "en": "Duval Pentagon uses H2, CH4, C2H6, C2H4 and C2H2, so it preserves more gas-composition information than the Duval Triangle. In this backend, duval_pentagon_fault is the consolidated result, while P1/P2 are retained separately.",
    },
}
def _method_knowledge(method: str, language: str) -> str:
    return METHOD_KNOWLEDGE[method][language]
def _asks_types(question: str) -> bool:
    return _contains_any(question, ["fault type", "fault types", "loại lỗi", "những lỗi", "gồm những lỗi", "có những loại", "types", "ประเภท", "มีอะไรบ้าง", "อะไรบ้าง"])
def _asks_explanation(question: str) -> bool:
    return _contains_any(question, ["vì sao", "tại sao", "nguyên nhân", "giải thích", "do đâu", "why", "reason", "explain", "cause", "ทำไม", "เพราะอะไร", "สาเหตุ", "อธิบาย"])
def _asks_comparison(question: str) -> bool:
    return _contains_any(question, ["so sánh", "khác nhau", "khác gì", "khác ở đâu", "giống nhau", "đối chiếu", "compare", "comparison", "difference", "different", "similar", "versus", "vs", "เปรียบเทียบ", "แตกต่าง", "ต่างกัน", "เหมือนกัน"])

# ============================================================
# FAST SQL
# ============================================================
def _open_ro():
    if not DB_PATH.exists(): raise FileNotFoundError("No DGA dataset loaded.")
    uri = f"file:{Path(DB_PATH).resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=3)
def _query(sql: str, params: tuple[Any, ...] = ()):
    conn = _open_ro()
    try:
        cur = conn.execute(sql, params)
        columns = [x[0] for x in (cur.description or [])]
        rows = cur.fetchmany(200)
        return columns, rows
    finally:
        conn.close()
def _fmt(value: Any) -> str:
    if value is None: return "N/A"
    try:
        n = float(value)
        if n.is_integer(): return str(int(n))
        return f"{n:.2f}"
    except (TypeError, ValueError):
        return str(value)

# ============================================================
# KNOWLEDGE HANDLERS
# ============================================================
def fast_domain_answer(question: str, language: str) -> str | None:
    methods = extract_requested_dga_methods(question)
    if _asks_comparison(question) and len(methods) >= 2:
        if language == "vi": lines = ["So sánh các phương pháp DGA:"]
        elif language == "th": lines = ["เปรียบเทียบวิธีวิเคราะห์ DGA:"]
        else: lines = ["Comparison of the requested DGA methods:"]
        for method, _ in methods:
            lines.append(f"- {method}: {_method_knowledge(method, language)}")
        if language == "vi":
            lines.append("Không có một phương pháp duy nhất luôn tốt nhất. Trong thực tế nên đối chiếu nhiều phương pháp vì chúng sử dụng các tập khí và cách biểu diễn khác nhau.")
        elif language == "th":
            lines.append("ไม่มีวิธีใดวิธีหนึ่งที่ดีที่สุดเสมอไป ควรพิจารณาหลายวิธีร่วมกัน เพราะแต่ละวิธีใช้ข้อมูลก๊าซและรูปแบบการวินิจฉัยต่างกัน")
        else:
            lines.append("No single method is always the best. Multiple methods should be compared because they use different gas information and diagnostic representations.")
        return "\n".join(lines)
    if len(methods) == 1 and _asks_types(question):
        method = methods[0][0]
        if method == "Duval Triangle":
            answers = {"vi": "Duval Triangle 1 có 6 fault type chính: PD, D1, D2, T1, T2 và T3.", "th": "Duval Triangle 1 มี fault type หลัก 6 ประเภท: PD, D1, D2, T1, T2 และ T3", "en": "Duval Triangle 1 has six main fault types: PD, D1, D2, T1, T2 and T3."}
            return answers[language]
        return _method_knowledge(method, language)
    if len(methods) == 1 and _contains_any(question, ["là gì", "nghĩa là gì", "hoạt động thế nào", "dùng như thế nào", "what is", "how does", "how it works", "meaning", "คืออะไร", "ทำงานอย่างไร", "หมายถึงอะไร"]):
        return _method_knowledge(methods[0][0], language)
    lowered = question.lower()
    if "c2h2" in lowered or "acetylene" in lowered:
        if _contains_any(question, ["là gì", "nghĩa là gì", "tại sao", "vì sao", "what is", "why", "meaning", "คืออะไร", "ทำไม"]):
            answers = {
                "vi": "C2H2 (acetylene) đặc biệt quan trọng trong DGA vì thường liên quan mạnh đến electrical discharge hoặc arcing. Không nên dùng C2H2 đơn độc để kết luận fault type; cần xem cùng C2H4, CH4, H2 và các phương pháp chẩn đoán khác.",
                "th": "C2H2 (acetylene) สำคัญมากใน DGA เพราะมักสัมพันธ์กับ electrical discharge หรือ arcing โดยเฉพาะเมื่อเพิ่มขึ้น ไม่ควรใช้ C2H2 เพียงตัวเดียวเพื่อตัดสิน fault type",
                "en": "C2H2 (acetylene) is particularly important in DGA because it is strongly associated with electrical discharge or arcing. It should not be used alone to determine a fault type.",
            }
            return answers[language]
    if "tdcg" in lowered and _contains_any(question, ["là gì", "nghĩa là gì", "what is", "meaning", "คืออะไร"]):
        answers = {
            "vi": "TDCG là tổng H2, CH4, C2H6, C2H4, C2H2 và CO. Nó phản ánh mức độ tạo khí cháy tổng thể nhưng không tự xác định fault type.",
            "th": "TDCG คือผลรวมของ H2, CH4, C2H6, C2H4, C2H2 และ CO ใช้สะท้อนระดับการเกิดก๊าซโดยรวมแต่ไม่ได้ระบุ fault type โดยตรง",
            "en": "TDCG is the sum of H2, CH4, C2H6, C2H4, C2H2 and CO. It reflects overall combustible-gas generation but does not identify a fault type by itself.",
        }
        return answers[language]
    if _contains_any(question, ["tất cả phương pháp", "tất cả các phương pháp", "toàn bộ phương pháp", "all methods", "all diagnostic methods", "ทุกวิธี", "ทั้ง 6 วิธี"]):
        names = ", ".join(ALL_DGA_METHODS)
        if language == "vi": return f"Backend hiện sử dụng 6 phương pháp DGA: {names}."
        if language == "th": return f"ระบบนี้ใช้วิธี DGA 6 วิธี ได้แก่ {names}"
        return f"The backend currently uses six DGA methods: {names}."
    return None

# ============================================================
# STRUCTURED DATABASE HANDLERS
# ============================================================
def _fast_count_critical(question: str, language: str) -> str | None:
    if not _contains_any(question, ["how many transformers are critical", "count critical transformers", "có bao nhiêu máy critical", "bao nhiêu máy đang critical", "critical กี่เครื่อง"]):
        return None
    _, rows = _query("SELECT COUNT(*) FROM transformers WHERE status='Critical'")
    count = rows[0][0] if rows else 0
    if language == "vi": return f"Hiện có {count} máy biến áp ở trạng thái Critical."
    if language == "th": return f"ขณะนี้มีหม้อแปลงสถานะ Critical จำนวน {count} เครื่อง"
    return f"There are currently {count} Critical transformers."
def _fast_most_critical(question: str, language: str) -> str | None:
    if not _contains_any(question, ["most critical transformer", "highest risk transformer", "highest priority transformer", "máy nào nghiêm trọng nhất", "máy biến áp nào nghiêm trọng nhất", "máy nào rủi ro cao nhất", "เครื่องไหนรุนแรงที่สุด", "หม้อแปลงตัวไหนเสี่ยงสูงสุด"]):
        return None
    columns, rows = _query("SELECT transformer_id,loc,status,fault_type,priority_score,rank,recommended_action FROM transformers ORDER BY rank ASC LIMIT 1")
    if not rows: return NO_DATA_REPLY[language]
    record = dict(zip(columns, rows[0]))
    if language == "vi":
        return f"Máy có mức ưu tiên cao nhất hiện tại là {record['transformer_id']}; status={record['status']}, fault={record['fault_type']}, priority_score={_fmt(record['priority_score'])}. Khuyến nghị: {record.get('recommended_action') or 'N/A'}"
    if language == "th":
        return f"หม้อแปลงที่มีความเสี่ยงสูงสุดคือ {record['transformer_id']}; status={record['status']}, fault={record['fault_type']}, priority_score={_fmt(record['priority_score'])}. คำแนะนำ: {record.get('recommended_action') or 'N/A'}"
    return f"The highest-priority transformer is {record['transformer_id']}; status={record['status']}, fault={record['fault_type']}, priority_score={_fmt(record['priority_score'])}. Recommendation: {record.get('recommended_action') or 'N/A'}"
def _fast_compare_transformers(question: str, language: str) -> str | None:
    if not _contains_any(question, ["so sánh", "compare", "comparison", "เปรียบเทียบ"]):
        return None
    ids = extract_transformer_ids(question)
    if len(ids) < 2: return None
    placeholders = ",".join("?" for _ in ids)
    sql = f"SELECT transformer_id,loc,status,fault_type,priority_score,rank,trend,recommended_action FROM transformers WHERE transformer_id IN ({placeholders}) ORDER BY rank ASC"
    columns, rows = _query(sql, tuple(ids))
    if not rows: return NO_DATA_REPLY[language]
    records = [dict(zip(columns, row)) for row in rows]
    if language == "vi": lines = ["So sánh các máy:"]
    elif language == "th": lines = ["ผลการเปรียบเทียบหม้อแปลง:"]
    else: lines = ["Transformer comparison:"]
    for record in records:
        lines.append(f"- {record['transformer_id']}: status={record['status']}, fault={record['fault_type']}, priority={_fmt(record['priority_score'])}, trend={record['trend']}, action={record['recommended_action'] or 'N/A'}")
    return "\n".join(lines)
def _fast_highest_gas(question: str, language: str) -> str | None:
    transformer_id = extract_transformer_id(question)
    if not transformer_id: return None
    if not _contains_any(question, ["khí nào", "nồng độ cao nhất", "highest gas", "highest concentration", "which gas", "highest gas concentration", "ก๊าซไหน", "ความเข้มข้นสูงสุด"]):
        return None
    gases = {"h2": "H2", "ch4": "CH4", "c2h6": "C2H6", "c2h4": "C2H4", "c2h2": "C2H2", "co": "CO", "co2": "CO2"}
    columns = list(gases.keys())
    rows = _query(f"SELECT sample_day,{','.join(columns)} FROM samples WHERE transformer_id=? ORDER BY sample_day ASC", (transformer_id,))[1]
    if not rows:
        return f"Không tìm thấy dữ liệu cho máy {transformer_id}." if language == "vi" else f"No data was found for transformer {transformer_id}." if language == "en" else f"ไม่พบข้อมูลของหม้อแปลง {transformer_id}"
    samples = []
    overall = {gas: [] for gas in gases}
    for row in rows:
        day = row[0]
        values = {}
        for i, gas in enumerate(gases, 1):
            value = row[i]
            try: value = float(value) if value is not None else None
            except (TypeError, ValueError): value = None
            values[gas] = value
            if value is not None: overall[gas].append(value)
        valid = {g: v for g, v in values.items() if v is not None}
        if valid:
            top_gas = max(valid, key=valid.get)
            samples.append((day, top_gas, valid[top_gas]))
    all_values = {gas: max(vals) for gas, vals in overall.items() if vals}
    if not all_values: return NO_DATA_REPLY[language]
    overall_gas = max(all_values, key=all_values.get)
    overall_value = all_values[overall_gas]
    if language == "vi":
        lines = [f"Máy {transformer_id}: khí có nồng độ cao nhất trong toàn bộ lịch sử là {gases[overall_gas]} ({_fmt(overall_value)} ppm).", "Theo từng sample:"]
    elif language == "th":
        lines = [f"หม้อแปลง {transformer_id}: ก๊าซที่มีความเข้มข้นสูงสุดในประวัติคือ {gases[overall_gas]} ({_fmt(overall_value)} ppm)", "แยกตาม sample:"]
    else:
        lines = [f"Transformer {transformer_id}: the highest recorded gas concentration is {gases[overall_gas]} ({_fmt(overall_value)} ppm).", "By sample:"]
    for day, gas, value in samples:
        lines.append(f"- {day}: {gases[gas]}={_fmt(value)} ppm")
    return "\n".join(lines)
def _fast_dga_method_history(question: str, language: str) -> str | None:
    transformer_id = extract_transformer_id(question)
    if not transformer_id: return None
    methods = extract_requested_dga_methods(question)
    if not methods: return None
    if not _contains_any(question, ["sample", "samples", "theo thời gian", "lịch sử", "history", "trend", "record", "results", "kết quả", "ตัวอย่าง", "ย้อนหลัง", "ตามเวลา"]):
        return None
    fields = ",".join(f'"{column}"' for _, column in methods)
    columns, rows = _query(f"SELECT sample_day,{fields} FROM samples WHERE transformer_id=? ORDER BY sample_day ASC", (transformer_id,))
    if not rows: return NO_DATA_REPLY[language]
    if language == "vi":
        lines = [f"Kết quả DGA của máy {transformer_id}:", "Phương pháp: " + ", ".join(x[0] for x in methods)]
    elif language == "th":
        lines = [f"ผล DGA ของหม้อแปลง {transformer_id}:", "วิธีวิเคราะห์: " + ", ".join(x[0] for x in methods)]
    else:
        lines = [f"DGA results for transformer {transformer_id}:", "Methods: " + ", ".join(x[0] for x in methods)]
    for row in rows:
        record = dict(zip(columns, row))
        parts = []
        for method, column in methods:
            value = record.get(column)
            parts.append(f"{method}={value if value not in (None, '') else 'ABSTAIN'}")
        lines.append(f"- {record['sample_day']}: " + "; ".join(parts))
    return "\n".join(lines)
def _fast_fault_history(question: str, language: str) -> str | None:
    transformer_id = extract_transformer_id(question)
    if not transformer_id: return None
    if not _contains_any(question, ["lỗi", "fault", "failure", "diagnosis", "diagnostic"]): return None
    asks_frequency = _contains_any(question, ["nhiều nhất", "phổ biến nhất", "xuất hiện nhiều", "most common", "most frequent", "highest frequency", "บ่อยที่สุด", "มากที่สุด"])
    asks_timeline = _contains_any(question, ["từng sample", "theo thời gian", "từng lần", "timeline", "over time", "each sample", "history", "ตามเวลา", "แต่ละ sample"])
    asks_why = _asks_explanation(question)
    if not (asks_frequency or asks_timeline or asks_why): return None
    columns, rows = _query("""
        WITH fault_counts AS (
            SELECT consensus_fault,
                   COUNT(*) AS sample_count,
                   AVG(severity_score) AS avg_severity,
                   MAX(severity_score) AS max_severity
            FROM samples
            WHERE transformer_id=?
              AND consensus_fault IS NOT NULL
              AND TRIM(consensus_fault)<>''
              AND UPPER(consensus_fault)<>'ABSTAIN'
            GROUP BY consensus_fault
        )
        SELECT consensus_fault,sample_count,avg_severity,max_severity
        FROM fault_counts
        ORDER BY sample_count DESC,avg_severity DESC
        """, (transformer_id,))
    if not rows:
        return f"Không tìm thấy dữ liệu cho máy {transformer_id}." if language == "vi" else f"ไม่พบข้อมูลของหม้อแปลง {transformer_id}" if language == "th" else f"No data was found for transformer {transformer_id}."
    fault_records = [dict(zip(columns, row)) for row in rows]
    top = fault_records[0]
    sample_columns, sample_rows = _query("""
        SELECT sample_day, consensus_fault, student_fault_label, diagnostic_confidence,
               severity_score, ieee_dga_status_label, h2,c2h2,c2h4,tdcg
        FROM samples
        WHERE transformer_id=?
        ORDER BY sample_day ASC
        """, (transformer_id,))
    samples = [dict(zip(sample_columns, row)) for row in sample_rows]
    total = len(samples)
    if language == "vi":
        lines = [f"Máy {transformer_id} có lỗi xuất hiện nhiều nhất là {top['consensus_fault']} ({top['sample_count']}/{total} sample)."]
    elif language == "th":
        lines = [f"หม้อแปลง {transformer_id} พบ fault ที่เกิดบ่อยที่สุดคือ {top['consensus_fault']} ({top['sample_count']}/{total} samples)"]
    else:
        lines = [f"Transformer {transformer_id} has the most frequent fault {top['consensus_fault']} ({top['sample_count']}/{total} samples)."]
    if asks_why:
        if language == "vi": lines.append(f"Severity trung bình của fault này là {_fmt(top['avg_severity'])}, cao nhất {_fmt(top['max_severity'])}.")
        elif language == "th": lines.append(f"severity เฉลี่ยของ fault นี้คือ {_fmt(top['avg_severity'])} และค่าสูงสุด {_fmt(top['max_severity'])}")
        else: lines.append(f"The average severity for this fault is {_fmt(top['avg_severity'])}, with a maximum of {_fmt(top['max_severity'])}.")
    if asks_timeline:
        if language == "vi": lines.append("Timeline từng sample:")
        elif language == "th": lines.append("ลำดับ sample ตามเวลา:")
        else: lines.append("Sample-by-sample timeline:")
        for record in samples:
            lines.append(f"- {record['sample_day']}: fault={record['consensus_fault'] or 'ABSTAIN'}, student={record['student_fault_label'] or 'N/A'}, severity={_fmt(record['severity_score'])}, IEEE={record['ieee_dga_status_label'] or 'N/A'}, H2={_fmt(record['h2'])}, C2H2={_fmt(record['c2h2'])}, TDCG={_fmt(record['tdcg'])}")
    return "\n".join(lines)

# ============================================================
# MAIN FAST-PATH
# ============================================================
def fast_path(question: str, context: dict | None = None) -> str | None:
    language = detect_language(question)
    result = fast_domain_answer(question, language)
    if result is not None: return result
    if not DB_PATH.exists(): return None
    handlers = (_fast_highest_gas, _fast_fault_history, _fast_count_critical, _fast_most_critical, _fast_compare_transformers, _fast_dga_method_history)
    for handler in handlers:
        try: result = handler(question, language)
        except Exception: continue
        if result is not None: return result
    return None
def obvious_out_of_scope(question: str) -> bool:
    text = (question or "").lower()
    return _contains_any(text, ["weather", "recipe", "football", "soccer score", "movie recommendation", "bitcoin price", "write python code", "debug my code"])