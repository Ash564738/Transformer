# seed_user.py
"""Sets the single dashboard login. There is no self-service registration —
run this once (and again any time you want to change the email/password)
to control who can sign in.

    python seed_user.py <email> <password> [name]

Re-running replaces whichever account exists and signs out any active
sessions for it.
"""
import sys

import auth


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python seed_user.py <email> <password> [name]")
        raise SystemExit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else email.split("@")[0]

    auth.init_db()
    try:
        user = auth.set_single_user(email, password, name)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"Login set: {user['email']} ({user['name']})")


if __name__ == "__main__":
    main()
