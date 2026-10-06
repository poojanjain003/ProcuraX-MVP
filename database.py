"""SQLite persistence for ProcuraX MVP.

Python's built-in sqlite3 module is used, so no separate database installation is required.
Includes persistent operational state plus demo user/access management.
"""
from __future__ import annotations
import hashlib, json, os, sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path(__file__).resolve().parent / "procurax.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return salt.hex() + "$" + digest.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return digest.hex() == digest_hex
    except Exception:
        return False


def init_db(default_users: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                manager_username TEXT NOT NULL,
                run_number INTEGER NOT NULL,
                run_timestamp TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                PRIMARY KEY (manager_username, run_number)
            );
            CREATE TABLE IF NOT EXISTS app_state (
                manager_username TEXT NOT NULL,
                state_key TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (manager_username, state_key)
            );
            CREATE TABLE IF NOT EXISTS workflow_state (
                manager_username TEXT NOT NULL,
                sku TEXT NOT NULL,
                workflow_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (manager_username, sku)
            );
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                category TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'manager',
                job_role TEXT,
                purchaser TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        if default_users:
            now = datetime.now().isoformat(timespec="seconds")
            for username, u in default_users.items():
                existing = conn.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone()
                if existing is None:
                    conn.execute("""INSERT INTO users
                        (username, display_name, password_hash, category, role, job_role, purchaser, enabled, created_at, updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (username, u["name"], _hash_password(u["password"]), u["category"], u.get("role","manager"),
                         u.get("job_role"), u.get("purchaser"), 1, now, now))


def list_users() -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT username, display_name, category, role, job_role, purchaser, enabled, created_at, updated_at FROM users ORDER BY role DESC, display_name").fetchall()
    return [dict(r) for r in rows]


def get_user(username: str) -> Optional[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username.lower().strip(),)).fetchone()
    return dict(row) if row else None


def user_exists(username: str) -> bool:
    return get_user(username) is not None


def authenticate_user(username: str, password: str) -> Optional[dict[str, Any]]:
    user = get_user(username)
    if not user or not bool(user["enabled"]):
        return None
    if not _verify_password(password, user["password_hash"]):
        return None
    return user


def create_user(username: str, display_name: str, password: str, category: str, role: str = "manager", job_role: str = "Procurement Manager", purchaser: str = "—") -> None:
    username = username.lower().strip()
    if not username or not display_name.strip() or not password:
        raise ValueError("Username, display name and password are required.")
    if not username.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Username may contain letters, numbers, hyphens and underscores only.")
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            raise ValueError("That username already exists.")
        conn.execute("""INSERT INTO users
            (username, display_name, password_hash, category, role, job_role, purchaser, enabled, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (username, display_name.strip(), _hash_password(password), category, role, job_role.strip(), purchaser.strip() or "—", 1, now, now))


def update_user(username: str, display_name: str, category: str, role: str, job_role: str, purchaser: str, enabled: bool, new_password: Optional[str] = None) -> None:
    username = username.lower().strip()
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            raise ValueError("User not found.")
        if new_password:
            conn.execute("""UPDATE users SET display_name=?, category=?, role=?, job_role=?, purchaser=?, enabled=?, password_hash=?, updated_at=? WHERE username=?""",
                         (display_name.strip(), category, role, job_role.strip(), purchaser.strip() or "—", int(enabled), _hash_password(new_password), now, username))
        else:
            conn.execute("""UPDATE users SET display_name=?, category=?, role=?, job_role=?, purchaser=?, enabled=?, updated_at=? WHERE username=?""",
                         (display_name.strip(), category, role, job_role.strip(), purchaser.strip() or "—", int(enabled), now, username))


def set_user_enabled(username: str, enabled: bool) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute("UPDATE users SET enabled=?, updated_at=? WHERE username=?", (int(enabled), now, username.lower().strip()))


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dump(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _load(value: str) -> Any:
    return json.loads(value)


def save_state(manager_username: str, state: Dict[str, Any]) -> None:
    init_db(); now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        for key, value in state.items():
            conn.execute("""INSERT INTO app_state(manager_username,state_key,state_json,updated_at) VALUES (?,?,?,?)
                ON CONFLICT(manager_username,state_key) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (manager_username, key, _dump(value), now))


def load_state(manager_username: str) -> Dict[str, Any]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT state_key,state_json FROM app_state WHERE manager_username=?", (manager_username,)).fetchall()
    return {r["state_key"]: _load(r["state_json"]) for r in rows}


def save_workflow(manager_username: str, workflow: Dict[str, Dict[str, Any]]) -> None:
    init_db(); now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute("DELETE FROM workflow_state WHERE manager_username=?", (manager_username,))
        for sku, payload in workflow.items():
            conn.execute("INSERT INTO workflow_state(manager_username,sku,workflow_json,updated_at) VALUES (?,?,?,?)", (manager_username,sku,_dump(payload),now))


def load_workflow(manager_username: str) -> Dict[str, Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT sku,workflow_json FROM workflow_state WHERE manager_username=?", (manager_username,)).fetchall()
    return {r["sku"]:_load(r["workflow_json"]) for r in rows}


def save_analysis_run(manager_username: str, run_number: int, payload: Dict[str, Any]) -> None:
    init_db(); now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO analysis_runs(manager_username,run_number,run_timestamp,analysis_json) VALUES (?,?,?,?)", (manager_username,int(run_number),str(payload.get("timestamp",now)),_dump(payload)))


def load_analysis_runs(manager_username: str) -> list[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT analysis_json FROM analysis_runs WHERE manager_username=? ORDER BY run_number DESC", (manager_username,)).fetchall()
    return [_load(r["analysis_json"]) for r in rows]


def clear_analysis_runs(manager_username: str) -> None:
    init_db()
    with _connect() as conn: conn.execute("DELETE FROM analysis_runs WHERE manager_username=?", (manager_username,))


def clear_manager_data(manager_username: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM app_state WHERE manager_username=?", (manager_username,))
        conn.execute("DELETE FROM workflow_state WHERE manager_username=?", (manager_username,))
        conn.execute("DELETE FROM analysis_runs WHERE manager_username=?", (manager_username,))
