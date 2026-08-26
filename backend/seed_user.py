# backend/seed_user.py

"""
Seed the local SQLite dashboard account.

Usage:

    python backend/seed_user.py

or:

    python backend/seed_user.py <email> <password> [name]

Important:
This script is for local SQLite mode.

For Vercel/serverless deployment, configure:
    DGA_ADMIN_EMAIL
    DGA_ADMIN_PASSWORD
    DGA_ADMIN_NAME
    DGA_AUTH_SECRET

and do not rely on users.db.
"""

import sys

import auth


DEFAULT_EMAIL = "admin@dga.local"
DEFAULT_PASSWORD = "admin123"
DEFAULT_NAME = "Administrator"


def main() -> None:
    if auth.is_stateless_mode():
        print(
            "Stateless authentication is enabled because "
            "DGA_AUTH_SECRET is configured."
        )
        print(
            "Do not seed users.db in deployment mode."
        )
        return

    auth.init_db()

    if len(sys.argv) == 1:
        email = DEFAULT_EMAIL
        password = DEFAULT_PASSWORD
        name = DEFAULT_NAME

        print(
            f"Seeding default account: "
            f"{email} / {password}"
        )

    elif len(sys.argv) >= 3:
        email = sys.argv[1]
        password = sys.argv[2]
        name = (
            sys.argv[3]
            if len(sys.argv) >= 4
            else email.split("@")[0]
        )

        print(
            f"Seeding custom account: {email}"
        )

    else:
        print(
            "Usage: "
            "python backend/seed_user.py "
            "[<email> <password> [name]]"
        )
        raise SystemExit(1)

    try:
        user = auth.set_single_user(
            email,
            password,
            name,
        )

    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(
        f"Login ready: "
        f"{user['email']} ({user['name']})"
    )


if __name__ == "__main__":
    main()