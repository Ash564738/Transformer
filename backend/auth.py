# backend/auth.py
"""
Authentication for the DGA dashboard.

Two modes are supported:

1. Local / traditional mode:
   - Uses backend/database/users.db
   - Compatible with seed_user.py
   - Sessions are stored in SQLite

2. Stateless deployment mode:
   - Activated automatically when DGA_AUTH_SECRET is set
   - No users.db or writable filesystem is required
   - The single configured account is read from environment variables:
       DGA_ADMIN_EMAIL
       DGA_ADMIN_PASSWORD
       DGA_ADMIN_NAME
       DGA_AUTH_SECRET
       DGA_SESSION_HOURS
   - Session tokens are signed with itsdangerous.

The stateless mode is intended for Vercel/serverless deployment.
"""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from config import DATABASE_DIR


DB_PATH = DATABASE_DIR / "users.db"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------

AUTH_SECRET = (os.getenv("DGA_AUTH_SECRET") or "").strip()

ADMIN_EMAIL = (
    os.getenv("DGA_ADMIN_EMAIL")
    or os.getenv("ADMIN_EMAIL")
    or ""
).strip().lower()

ADMIN_PASSWORD = (
    os.getenv("DGA_ADMIN_PASSWORD")
    or os.getenv("ADMIN_PASSWORD")
    or ""
)

ADMIN_NAME = (
    os.getenv("DGA_ADMIN_NAME")
    or os.getenv("ADMIN_NAME")
    or "Administrator"
).strip()

try:
    SESSION_HOURS = max(
        1,
        int(os.getenv("DGA_SESSION_HOURS", "24")),
    )
except ValueError:
    SESSION_HOURS = 24


def is_stateless_mode() -> bool:
    """
    Vercel deployment mode is enabled when DGA_AUTH_SECRET is configured.
    """
    return bool(AUTH_SECRET)


# ---------------------------------------------------------------------------
# SQLite mode
# ---------------------------------------------------------------------------

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the SQLite authentication database.

    In stateless deployment mode there is deliberately no database
    initialization because Vercel filesystem state should not be used
    as persistent authentication storage.
    """
    if is_stateless_mode():
        return

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
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
    }


# ---------------------------------------------------------------------------
# Stateless account configuration
# ---------------------------------------------------------------------------

def _validate_stateless_config() -> None:
    """
    Validate environment variables required by deployment mode.
    """
    if not AUTH_SECRET:
        raise RuntimeError(
            "DGA_AUTH_SECRET is not configured."
        )

    if not ADMIN_EMAIL:
        raise RuntimeError(
            "DGA_ADMIN_EMAIL is not configured."
        )

    if not EMAIL_RE.match(ADMIN_EMAIL):
        raise RuntimeError(
            "DGA_ADMIN_EMAIL is not a valid email address."
        )

    if not ADMIN_PASSWORD:
        raise RuntimeError(
            "DGA_ADMIN_PASSWORD is not configured."
        )

    if not ADMIN_NAME:
        raise RuntimeError(
            "DGA_ADMIN_NAME is not configured."
        )


def _stateless_user() -> dict:
    """
    Return the configured deployment user.
    """
    _validate_stateless_config()

    return {
        "id": 1,
        "email": ADMIN_EMAIL,
        "name": ADMIN_NAME,
    }


def _serializer() -> URLSafeTimedSerializer:
    """
    Build the signed-token serializer.
    """
    _validate_stateless_config()

    return URLSafeTimedSerializer(
        AUTH_SECRET,
        salt="dga-dashboard-auth-v1",
    )


def _max_age_seconds() -> int:
    return SESSION_HOURS * 60 * 60


# ---------------------------------------------------------------------------
# Seed/local-user functions
# ---------------------------------------------------------------------------

def set_single_user(
    email: str,
    password: str,
    name: str,
) -> dict:
    """
    Replace the local SQLite account with exactly one account.

    This function is intentionally disabled in stateless deployment mode.
    """
    if is_stateless_mode():
        raise RuntimeError(
            "set_single_user() is disabled in stateless deployment mode. "
            "Configure DGA_ADMIN_EMAIL, DGA_ADMIN_PASSWORD and "
            "DGA_ADMIN_NAME instead."
        )

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
            """
            INSERT INTO users
                (email, name, password_hash, created_at)
            VALUES
                (?, ?, ?, ?)
            """,
            (
                email,
                name,
                generate_password_hash(password),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        conn.commit()

        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()

        return _user_public(row)

    finally:
        conn.close()


def has_user() -> bool:
    """
    Check whether a local SQLite user exists.
    """
    if is_stateless_mode():
        return bool(ADMIN_EMAIL and ADMIN_PASSWORD)

    conn = _connect()

    try:
        return (
            conn.execute(
                "SELECT 1 FROM users LIMIT 1"
            ).fetchone()
            is not None
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

def _create_stateless_session(user: dict) -> str:
    """
    Create a signed stateless session token.
    """
    serializer = _serializer()

    payload = {
        "type": "dga-session",
        "user": user,
    }

    signed = serializer.dumps(payload)

    return f"stateless.{signed}"


def create_session(user_id: int) -> str:
    """
    Create a local SQLite session.

    In stateless mode, user_id is ignored and the configured deployment
    account is used.
    """
    if is_stateless_mode():
        return _create_stateless_session(_stateless_user())

    token = secrets.token_urlsafe(32)

    conn = _connect()

    try:
        conn.execute(
            """
            INSERT INTO sessions
                (token, user_id, created_at)
            VALUES
                (?, ?, ?)
            """,
            (
                token,
                user_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        conn.commit()

    finally:
        conn.close()

    return token


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _stateless_login(
    email: str,
    password: str,
) -> tuple[dict, str]:
    """
    Authenticate against deployment environment variables.
    """
    user = _stateless_user()

    normalized_email = (email or "").strip().lower()

    if normalized_email != user["email"]:
        raise ValueError("Incorrect email or password.")

    if not secrets.compare_digest(
        password or "",
        ADMIN_PASSWORD,
    ):
        raise ValueError("Incorrect email or password.")

    token = _create_stateless_session(user)

    return user, token


def _sqlite_login(
    email: str,
    password: str,
) -> tuple[dict, str]:
    """
    Authenticate against the local SQLite database.
    """
    normalized_email = (email or "").strip().lower()

    conn = _connect()

    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()

    finally:
        conn.close()

    if not row:
        raise ValueError("Incorrect email or password.")

    if not check_password_hash(
        row["password_hash"],
        password or "",
    ):
        raise ValueError("Incorrect email or password.")

    token = create_session(row["id"])

    return _user_public(row), token


def login(
    email: str,
    password: str,
) -> tuple[dict, str]:
    """
    Authenticate the configured dashboard account.
    """
    if is_stateless_mode():
        return _stateless_login(email, password)

    return _sqlite_login(email, password)


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------

def _user_from_stateless_token(token: str) -> dict | None:
    """
    Validate a signed stateless token.
    """
    if not token.startswith("stateless."):
        return None

    encoded = token[len("stateless."):]

    if not encoded:
        return None

    try:
        serializer = _serializer()

        payload = serializer.loads(
            encoded,
            max_age=_max_age_seconds(),
        )

    except SignatureExpired:
        return None

    except BadSignature:
        return None

    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    if payload.get("type") != "dga-session":
        return None

    user = payload.get("user")

    if not isinstance(user, dict):
        return None

    if user.get("email") != ADMIN_EMAIL:
        return None

    return {
        "id": user.get("id", 1),
        "email": user["email"],
        "name": user.get("name", ADMIN_NAME),
    }


def _user_from_sqlite_token(token: str) -> dict | None:
    """
    Validate a local SQLite session token.
    """
    if not token:
        return None

    conn = _connect()

    try:
        row = conn.execute(
            """
            SELECT users.*
            FROM sessions
            JOIN users
                ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()

    finally:
        conn.close()

    return _user_public(row) if row else None


def user_from_token(token: str) -> dict | None:
    """
    Resolve a client token into a public user object.
    """
    if not token:
        return None

    if is_stateless_mode():
        return _user_from_stateless_token(token)

    return _user_from_sqlite_token(token)


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

def logout(token: str) -> None:
    """
    Logout.

    Stateless tokens cannot be revoked individually because there is no
    persistent session store. They expire automatically after
    DGA_SESSION_HOURS.

    Local SQLite sessions are explicitly deleted.
    """
    if is_stateless_mode():
        return

    if not token:
        return

    conn = _connect()

    try:
        conn.execute(
            "DELETE FROM sessions WHERE token = ?",
            (token,),
        )
        conn.commit()

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Flask helpers
# ---------------------------------------------------------------------------

def _extract_token() -> str | None:
    header = request.headers.get("Authorization", "")

    if not header.startswith("Bearer "):
        return None

    token = header[len("Bearer "):].strip()

    return token or None


def require_auth(fn):
    """
    Flask decorator requiring a valid authentication token.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _extract_token()

        user = (
            user_from_token(token)
            if token
            else None
        )

        if not user:
            return jsonify(
                error="Authentication required."
            ), 401

        request.current_user = user

        return fn(*args, **kwargs)

    return wrapper