# components/theme.py
import streamlit as st

def apply_theme():
    st.set_page_config(
        page_title="Transformer Degradation Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            --bg-app: #050505;
            --bg-app-2: #0a0a0a;
            --bg-panel: #0f0f10;
            --bg-panel-2: #141416;
            --bg-panel-3: #1a1a1d;
            --bg-hover: #18181b;

            --border: #242428;
            --border-2: #2d2d31;
            --border-soft: rgba(255,255,255,0.06);

            --text-1: #f5f5f7;
            --text-2: #d0d0d5;
            --text-3: #9a9aa3;
            --text-4: #7b7b84;

            --accent: #f5f5f7;
            --accent-soft: rgba(255,255,255,0.10);

            --success: #32d583;
            --warning: #f5b942;
            --danger: #ff5d5d;
            --info: #8ab4ff;

            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 18px;

            --shadow-1: 0 10px 30px rgba(0, 0, 0, 0.28);
            --shadow-2: 0 18px 50px rgba(0, 0, 0, 0.36);
        }

        html, body, [class*="css"] {
            color: var(--text-1);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.035), transparent 24%),
                radial-gradient(circle at bottom left, rgba(255,255,255,0.02), transparent 20%),
                linear-gradient(180deg, var(--bg-app) 0%, var(--bg-app-2) 100%);
            color: var(--text-1);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1680px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #070707 0%, #0b0b0c 100%);
            border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-bottom: 1.25rem;
        }

        /* Typography */
        h1, h2, h3, h4 {
            color: var(--text-1);
            letter-spacing: -0.01em;
        }

        p, label, .stCaption, .stMarkdown, .stText {
            color: var(--text-2);
        }

        /* Generic cards / containers */
        .panel-card {
            background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-1);
        }

        /* Metrics */
        .stMetric {
            background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
            padding: 0.95rem 1rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-1);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-1);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-3);
        }

        /* Inputs */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        textarea,
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: var(--text-1) !important;
            box-shadow: none !important;
        }

        textarea::placeholder,
        input::placeholder {
            color: var(--text-4) !important;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid var(--border-2);
            background: linear-gradient(180deg, #151517 0%, #101012 100%);
            color: var(--text-1);
            font-weight: 600;
            min-height: 2.7rem;
            transition: all 0.16s ease;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #3a3a40;
            background: linear-gradient(180deg, #1a1a1d 0%, #141417 100%);
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #f5f5f7 0%, #dcdce0 100%);
            color: #0a0a0b;
            border-color: #d7d7dc;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(180deg, #ffffff 0%, #e7e7eb 100%);
            border-color: #ffffff;
        }

        /* Radio / toggles / pills */
        div[role="radiogroup"] label {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
        }

        /* Expander / alerts */
        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
            border-radius: 14px;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: var(--bg-panel);
        }

        /* Tabs */
        button[role="tab"] {
            border-radius: 12px 12px 0 0;
        }

        /* Chat shell */
        .chat-shell {
            margin-top: 0.35rem;
            margin-bottom: 1rem;
            padding: 1rem 1.1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            background: linear-gradient(180deg, #0d0d0f 0%, #111113 100%);
            box-shadow: var(--shadow-1);
        }

        .chat-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .chat-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--text-1);
        }

        .chat-subtitle {
            margin-top: 0.25rem;
            color: var(--text-3);
            font-size: 0.9rem;
        }

        .chat-context-pill {
            padding: 0.45rem 0.72rem;
            border: 1px solid var(--border-2);
            border-radius: 999px;
            background: var(--bg-panel-3);
            color: var(--text-2);
            font-size: 0.85rem;
            white-space: nowrap;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            background: linear-gradient(180deg, #0e0e10 0%, #131316 100%);
            border: 1px solid var(--border);
            padding: 0.45rem 0.55rem;
        }

        /* Custom cards used in results */
        .dg-card {
            background: linear-gradient(180deg, #0f0f10 0%, #151518 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: var(--shadow-1);
        }

        .dg-card-accent {
            width: 100%;
            height: 4px;
            border-radius: 999px;
            margin-bottom: 12px;
            background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0.15));
        }

        .dg-card-title {
            font-size: 0.82rem;
            color: var(--text-3);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .dg-card-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text-1);
            margin-top: 8px;
            line-height: 1.15;
        }

        .dg-card-subtitle {
            font-size: 0.92rem;
            color: var(--text-2);
            margin-top: 6px;
        }

        .dg-card-red .dg-card-accent {
            background: linear-gradient(90deg, #ff6b6b, rgba(255,107,107,0.15));
        }

        .dg-card-amber .dg-card-accent {
            background: linear-gradient(90deg, #f5b942, rgba(245,185,66,0.15));
        }

        .dg-card-green .dg-card-accent {
            background: linear-gradient(90deg, #32d583, rgba(50,213,131,0.15));
        }

        .dg-card-blue .dg-card-accent {
            background: linear-gradient(90deg, #8ab4ff, rgba(138,180,255,0.15));
        }

        .insight-box {
            background: linear-gradient(180deg, #0f0f10 0%, #151518 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
            margin: 10px 0 4px;
        }

        .insight-box-title {
            font-weight: 700;
            color: var(--text-1);
            margin-bottom: 0.35rem;
        }

        .insight-box-body {
            color: var(--text-2);
            font-size: 0.95rem;
            line-height: 1.55;
        }

        hr {
            border-color: var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )