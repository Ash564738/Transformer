# auth.py
"""Minimal single-user login for the DGA dashboard.

Local SQLite store (backend/data/users.db) — no external auth provider
needed for a single internal user. Passwords are hashed with werkzeug's
PBKDF2 helper (already a Flask dependency, no extra package). Sessions are
opaque random tokens stored server-side, sent by the client as
`Authorization: Bearer <token>`.

There is no self-service registration: the `users` table holds at most one
row, set with `python seed_user.py <email> <password> [name]`. Re-running
the seed script replaces whichever account exists and invalidates its
existing sessions.
"""
import sqlite3
import secrets
import re
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).resolve().parent / "data" / "users.db"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _user_public(row) -> dict:
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


def set_single_user(email: str, password: str, name: str) -> dict:
    """Replaces whichever account exists with exactly one (email, password,
    name). Used by seed_user.py, not exposed over HTTP — there is no
    self-service registration for this single-user dashboard."""
    email = (email or "").strip().lower()
    name = (name or "").strip()
    if not EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address.")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")
    if not name:
        raise ValueError("Name is required.")

    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")
        cur = conn.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (email, name, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _user_public(row)
    finally:
        conn.close()


def has_user() -> bool:
    conn = _connect()
    try:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None
    finally:
        conn.close()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def login(email: str, password: str) -> tuple[dict, str]:
    email = (email or "").strip().lower()
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    if not row or not check_password_hash(row["password_hash"], password or ""):
        raise ValueError("Incorrect email or password.")
    token = create_session(row["id"])
    return _user_public(row), token


def user_from_token(token: str) -> dict | None:
    if not token:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
    finally:
        conn.close()
    return _user_public(row) if row else None


def logout(token: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def _extract_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer "):].strip()
    return None


def require_auth(fn):
    """Route decorator: 401s unless a valid session token is present."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _extract_token()
        user = user_from_token(token) if token else None
        if not user:
            return jsonify(error="Authentication required."), 401
        request.current_user = user
        return fn(*args, **kwargs)
    return wrapper
