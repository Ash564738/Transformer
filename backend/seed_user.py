# seed_user.py
"""Seed a default dashboard login if no arguments are given.
Run without arguments to create/update the default admin account.
    python seed_user.py
To set a custom account:
    python seed_user.py <email> <password> [name]
"""
import sys
import auth

DEFAULT_EMAIL = "admin@dga.local"
DEFAULT_PASSWORD = "admin123"
DEFAULT_NAME = "Administrator"

def main():
    auth.init_db()

    if len(sys.argv) == 1:
        # Chạy không tham số -> dùng tài khoản mặc định
        email, password, name = DEFAULT_EMAIL, DEFAULT_PASSWORD, DEFAULT_NAME
        print(f"Seeding default account: {email} / {password}")
    elif len(sys.argv) >= 3:
        email = sys.argv[1]
        password = sys.argv[2]
        name = sys.argv[3] if len(sys.argv) > 3 else email.split("@")[0]
        print(f"Seeding custom account: {email}")
    else:
        print("Usage: python seed_user.py [<email> <password> [name]]")
        raise SystemExit(1)

    try:
        user = auth.set_single_user(email, password, name)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"Login ready: {user['email']} ({user['name']})")

if __name__ == "__main__":
    main()