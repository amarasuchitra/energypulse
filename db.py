"""
Database Management for EnergyPulse
===================================
Persistent storage layer using SQLite for:
  - Users (email-based)
  - Family members (linked to primary user)
  - Notification preferences (per family member)
  - Notification log (audit trail)
  - Chatbot conversation history

All data is encrypted at rest and accessed via context managers.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import hashlib
import secrets
import pandas as pd


DB_PATH = "energypulse.db"


class DatabaseManager:
    """
    Centralized database access with automatic schema initialization.
    Thread-safe via context manager pattern.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema on first run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    household_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_guest BOOLEAN DEFAULT 0,
                    preferred_language TEXT DEFAULT 'en',
                    home_type TEXT,
                    occupants INTEGER,
                    tariff_rate REAL,
                    city TEXT,
                    home_size INTEGER,
                    appliances_json TEXT
                )
            """)
            
            # Family members table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS family_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    household_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    relationship TEXT,
                    email TEXT NOT NULL,
                    phone TEXT,
                    preferred_language TEXT DEFAULT 'en',
                    notification_bill_alerts BOOLEAN DEFAULT 1,
                    notification_optimization_tips BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (household_id) REFERENCES users(household_id),
                    UNIQUE(household_id, email)
                )
            """)
            
            # Notifications log (audit trail)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notifications_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    household_id TEXT NOT NULL,
                    recipient_email TEXT NOT NULL,
                    recipient_name TEXT,
                    notification_type TEXT NOT NULL,
                    triggered_by TEXT,
                    trigger_data_json TEXT,
                    subject TEXT NOT NULL,
                    body_text TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'sent',
                    error_message TEXT,
                    FOREIGN KEY (household_id) REFERENCES users(household_id)
                )
            """)
            
            # Chatbot conversation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chatbot_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    household_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    question TEXT,
                    answer TEXT,
                    tool_calls_json TEXT,
                    grounding_data_json TEXT,
                    is_valid_grounded BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (household_id) REFERENCES users(household_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS household_settings (
                    household_id TEXT PRIMARY KEY,
                    bill_threshold REAL DEFAULT 1500.0,
                    last_bill_alert_at TIMESTAMP,
                    last_weekly_summary_at TIMESTAMP,
                    last_optimization_key TEXT,
                    onboarding_completed BOOLEAN DEFAULT 0,
                    onboarding_completed_at TIMESTAMP,
                    FOREIGN KEY (household_id) REFERENCES users(household_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS household_homes (
                    household_id TEXT PRIMARY KEY,
                    details_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    household_id TEXT NOT NULL,
                    datetime TIMESTAMP NOT NULL,
                    Global_active_power REAL NOT NULL,
                    Global_reactive_power REAL,
                    Voltage REAL,
                    Global_intensity REAL,
                    Sub_metering_1 REAL,
                    Sub_metering_2 REAL,
                    Sub_metering_3 REAL,
                    hour INTEGER,
                    day_of_week INTEGER,
                    is_weekend INTEGER,
                    day_of_month INTEGER,
                    week_of_year INTEGER,
                    FOREIGN KEY (household_id) REFERENCES users(household_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_history_household_time ON simulation_history (household_id, datetime)")

            conn.commit()

    # ────────────── REMOVING A HOUSEHOLD (guest sign-out, stale guests) ────────
    def delete_household(self, household_id: str, email: Optional[str] = None,
                         remove_files: bool = True) -> int:
        """
        Remove everything stored for one household: every table row keyed by
        its household_id, its user and credential rows, and (by default) its
        encrypted bill and appliance-photo files.  Returns rows deleted.
        """
        if not household_id:
            return 0
        deleted = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if email is None:
                row = cursor.execute("SELECT email FROM users WHERE household_id = ?",
                                     (household_id,)).fetchone()
                email = row["email"] if row else None
            tables = [r["name"] for r in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
            for table in tables:
                cols = {c["name"] for c in cursor.execute(f'PRAGMA table_info("{table}")')}
                if "household_id" in cols:
                    cursor.execute(f'DELETE FROM "{table}" WHERE household_id = ?', (household_id,))
                    deleted += max(cursor.rowcount, 0)
            for table in ("credentials", "login_tokens"):
                if email and table in tables:
                    cursor.execute(f'DELETE FROM "{table}" WHERE email = ?', (email,))
                    deleted += max(cursor.rowcount, 0)
        if remove_files:
            self._delete_household_files(household_id)
        return deleted

    @staticmethod
    def _delete_household_files(household_id: str) -> None:
        import re
        import shutil
        dirs = []
        try:
            from integrity import BILL_DIR
            if re.fullmatch(r"[A-Za-z0-9_.-]+", household_id) and household_id not in (".", ".."):
                dirs.append(os.path.join(BILL_DIR, household_id))
        except Exception:
            pass
        try:
            from nameplate import PHOTO_DIR
            safe = re.sub(r"[^A-Za-z0-9_-]", "", household_id)[:64]
            if safe:
                dirs.append(os.path.join(PHOTO_DIR, safe))
        except Exception:
            pass
        for d in dirs:
            if os.path.basename(d) and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    def purge_stale_guests(self, max_age_hours: float = 24) -> int:
        """Remove guest households older than max_age_hours (guests who never logged out)."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT household_id, email FROM users WHERE is_guest = 1 "
                "AND created_at < datetime('now', ?)", (f"-{float(max_age_hours)} hours",)).fetchall()
        for r in rows:
            self.delete_household(r["household_id"], r["email"])
        return len(rows)

    # ────────────── SMALL PER-HOUSEHOLD SETTINGS (goal, schedules, outages...) ───
    def get_store(self, household_id: str, key: str, default=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS household_store (
                household_id TEXT NOT NULL, key TEXT NOT NULL, value_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (household_id, key))""")
            cursor.execute("SELECT value_json FROM household_store WHERE household_id = ? AND key = ?",
                           (household_id, key))
            row = cursor.fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except (ValueError, TypeError):
            return default

    def set_store(self, household_id: str, key: str, value) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS household_store (
                household_id TEXT NOT NULL, key TEXT NOT NULL, value_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (household_id, key))""")
            cursor.execute("""INSERT INTO household_store (household_id, key, value_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(household_id, key) DO UPDATE SET
                    value_json = excluded.value_json, updated_at = CURRENT_TIMESTAMP""",
                           (household_id, key, json.dumps(value)))

    # ────────────── HOME SETUP (so a returning user lands on their own home) ─────
    def save_home_details(self, household_id: str, details: Dict[str, Any]) -> None:
        with self.get_connection() as conn:
            conn.cursor().execute("""
                INSERT INTO household_homes (household_id, details_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(household_id) DO UPDATE SET
                    details_json = excluded.details_json, updated_at = CURRENT_TIMESTAMP
            """, (household_id, json.dumps(details)))

    def get_home_details(self, household_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT details_json FROM household_homes WHERE household_id = ?",
                           (household_id,))
            row = cursor.fetchone()
        if not row:
            return None
        try:
            details = json.loads(row["details_json"])
        except (ValueError, TypeError):
            return None
        if not isinstance(details, dict):
            return None
        for key in ("peak_morning", "peak_evening"):       # JSON turns tuples into lists
            if isinstance(details.get(key), list):
                details[key] = tuple(details[key])
        return details

    # ────────────── SIMULATION HISTORY ──────────────────────────────
    def save_simulation_history(self, household_id: str, df) -> int:
        """Save replayed simulation data to database. Returns rows inserted."""
        if df is None or df.empty:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Prepare data - ensure required columns exist
            required_cols = ["datetime", "Global_active_power"]
            for col in required_cols:
                if col not in df.columns:
                    return 0
            
            # Select only columns that exist in the table
            cols = ["datetime", "Global_active_power", "Global_reactive_power", 
                    "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", 
                    "Sub_metering_3", "hour", "day_of_week", "is_weekend", 
                    "day_of_month", "week_of_year"]
            available_cols = [c for c in cols if c in df.columns]
            
            data_to_insert = df[available_cols].copy()
            # The caller passes the whole replay so far; keep only readings newer
            # than what is already stored, so nothing is saved twice.
            cursor.execute("SELECT MAX(datetime) AS last FROM simulation_history WHERE household_id = ?",
                           (household_id,))
            last = cursor.fetchone()["last"]
            data_to_insert["datetime"] = pd.to_datetime(data_to_insert["datetime"])
            if last is not None:
                data_to_insert = data_to_insert[data_to_insert["datetime"] > pd.to_datetime(last)]
            if data_to_insert.empty:
                return 0
            data_to_insert["datetime"] = data_to_insert["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            data_to_insert["household_id"] = household_id
            
            # Build insert statement
            placeholders = ", ".join(["?"] * len(available_cols + ["household_id"]))
            col_names = ", ".join(available_cols + ["household_id"])
            
            rows = [tuple(row) for row in data_to_insert.itertuples(index=False, name=None)]
            cursor.executemany(f"INSERT INTO simulation_history ({col_names}) VALUES ({placeholders})", rows)
        return len(rows)

    def get_simulation_history(self, household_id: str, limit: int = None) -> pd.DataFrame:
        """Load simulation history for a household."""
        with self.get_connection() as conn:
            query = "SELECT * FROM simulation_history WHERE household_id = ? ORDER BY datetime"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql_query(query, conn, params=(household_id,), parse_dates=["datetime"])
        return df

    # ────────────── USER MANAGEMENT ──────────────────────────────
    # Password hashing constants (matching auth.py)
    PBKDF2_ROUNDS = 310_000
    PBKDF2_ALGO = "sha256"

    @staticmethod
    def _hash_password(password: str, salt: bytes, algo: str = "sha256", rounds: int = 310_000) -> str:
        return hashlib.pbkdf2_hmac(algo, password.encode(), salt, rounds).hex()

    def create_user(self, email: str, name: str, password: str, is_guest: bool = False,
                   preferred_language: str = "en", household_id: Optional[str] = None) -> bool:
        """Create a new user account. Returns True on success."""
        email = (email or "").strip().lower()
        if household_id is None:
            household_id = hashlib.md5(email.encode()).hexdigest()
        
        salt = secrets.token_bytes(16)
        password_hash = self._hash_password(password, salt, self.PBKDF2_ALGO, self.PBKDF2_ROUNDS)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users 
                    (email, name, password_hash, salt, household_id, is_guest, preferred_language)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (email, name, password_hash, salt.hex(), household_id, is_guest, preferred_language))
            self.ensure_household_settings(household_id)
            return True
        except sqlite3.IntegrityError:
            return False  # Email already exists
    
    def ensure_user_login(self, email: str, name: str = None, is_guest: bool = False,
                          preferred_language: str = "en") -> Dict[str, Any]:
        """
        Sync an app login to the persistent user table (INSERT OR IGNORE),
        so notifications / family management work for in-memory-login users.
        Returns the user dict (with household_id).
        """
        email = (email or "").strip().lower()
        name = (name or "") or email.split("@")[0]
        household_id = hashlib.md5(email.encode()).hexdigest()
        salt = secrets.token_bytes(16)
        password_hash = self._hash_password((email + household_id), salt, self.PBKDF2_ALGO, self.PBKDF2_ROUNDS)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users
                (email, name, password_hash, salt, household_id, is_guest, preferred_language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    name = excluded.name,
                    is_guest = excluded.is_guest,
                    preferred_language = excluded.preferred_language
            """, (email, name, password_hash, salt.hex(), household_id,
                  1 if is_guest else 0, preferred_language))
        self.ensure_household_settings(household_id)
        return {
            "email": email,
            "name": name,
            "household_id": household_id,
            "guest": bool(is_guest),
            "language": preferred_language,
        }

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user. Returns user dict on success, None on failure."""
        email = (email or "").strip().lower()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email, name, household_id, is_guest, preferred_language, password_hash, salt
                FROM users 
                WHERE email = ?
            """, (email,))
            row = cursor.fetchone()
            
        if row:
            stored_hash = row["password_hash"]
            salt = bytes.fromhex(row["salt"])
            if self._hash_password(password, salt, self.PBKDF2_ALGO, self.PBKDF2_ROUNDS) == stored_hash:
                return {
                    "email": row["email"],
                    "name": row["name"],
                    "household_id": row["household_id"],
                    "guest": bool(row["is_guest"]),
                    "language": row["preferred_language"],
                }
        return None
    
    def get_user(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user details (no password check)."""
        email = (email or "").strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users WHERE email = ?
            """, (email,))
            row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def update_user_profile(self, email: str, updates: Dict[str, Any]) -> bool:
        """Update user profile (home details, language, etc.)."""
        allowed_fields = {
            'name', 'preferred_language', 'home_type', 'occupants',
            'tariff_rate', 'city', 'home_size', 'appliances_json'
        }
        
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        if not updates:
            return True
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [email]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE users SET {set_clause} WHERE email = ?", values)
        
        return True
    
    # ────────────── FAMILY MEMBER MANAGEMENT ──────────────────────────────
    
    def add_family_member(self, household_id: str, name: str, relationship: str,
                         email: str, phone: Optional[str] = None,
                         preferred_language: str = "en",
                         notify_bills: bool = True, notify_tips: bool = True) -> Tuple[bool, str]:
        """
        Add a family member to the household.
        Returns (success: bool, message: str)
        """
        email = (email or "").strip().lower()
        name = (name or "").strip()
        if not name:
            return (False, "missing_name")
        if not self._validate_email(email):
            return (False, "invalid_email")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM family_members 
                WHERE household_id = ? AND lower(email) = ?
            """, (household_id, email))
            if cursor.fetchone():
                return (False, "duplicate_email")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO family_members
                    (household_id, name, relationship, email, phone, preferred_language,
                     notification_bill_alerts, notification_optimization_tips)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (household_id, name, relationship, email, phone, preferred_language,
                     notify_bills, notify_tips))
            return (True, "member_added")
        except sqlite3.IntegrityError:
            return (False, "duplicate_member")
    
    def get_family_members(self, household_id: str) -> List[Dict[str, Any]]:
        """Get all family members for a household."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM family_members 
                WHERE household_id = ?
                ORDER BY created_at ASC
            """, (household_id,))
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def update_family_member(self, member_id: int, household_id: str,
                             updates: Dict[str, Any]) -> Tuple[bool, str]:
        """Update a family member's details. Scoped to household_id."""
        allowed_fields = {
            'name', 'relationship', 'email', 'phone', 'preferred_language',
            'notification_bill_alerts', 'notification_optimization_tips'
        }
        
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        if not updates:
            return (True, "member_updated")

        if 'email' in updates:
            updates['email'] = (updates['email'] or "").strip().lower()
            if not self._validate_email(updates['email']):
                return (False, "invalid_email")
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM family_members
                    WHERE household_id = ? AND lower(email) = ? AND id != ?
                """, (household_id, updates['email'], member_id))
                if cursor.fetchone():
                    return (False, "duplicate_email")

        if 'name' in updates:
            updates['name'] = (updates['name'] or "").strip()
            if not updates['name']:
                return (False, "missing_name")

        for bool_key in ('notification_bill_alerts', 'notification_optimization_tips'):
            if bool_key in updates:
                updates[bool_key] = 1 if updates[bool_key] else 0
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [member_id, household_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE family_members SET {set_clause} WHERE id = ? AND household_id = ?",
                values,
            )
            if cursor.rowcount == 0:
                return (False, "not_found")
        
        return (True, "member_updated")
    
    def remove_family_member(self, member_id: int, household_id: str) -> bool:
        """
        Remove a family member (immediately stops notifications).
        Only the correct household can remove their own members.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM family_members 
                WHERE id = ? AND household_id = ?
            """, (member_id, household_id))
            return cursor.rowcount > 0

    def family_member_exists(self, household_id: str, email: str) -> bool:
        """True only if this email is still a current family member of the household."""
        email = (email or "").strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM family_members
                WHERE household_id = ? AND lower(email) = ?
            """, (household_id, email))
            return cursor.fetchone() is not None
    
    # ────────────── NOTIFICATION MANAGEMENT ──────────────────────────────
    
    def log_notification(self, household_id: str, recipient_email: str,
                        recipient_name: str, notification_type: str,
                        triggered_by: str, trigger_data: Dict[str, Any],
                        subject: str, body_text: str, language: str = "en",
                        status: str = "sent", error_message: Optional[str] = None) -> int:
        """
        Log a notification (audit trail).
        Returns notification ID.
        """
        trigger_data_json = json.dumps(trigger_data) if trigger_data else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notifications_log
                (household_id, recipient_email, recipient_name, notification_type,
                 triggered_by, trigger_data_json, subject, body_text, language,
                 status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (household_id, recipient_email, recipient_name, notification_type,
                 triggered_by, trigger_data_json, subject, body_text, language,
                 status, error_message))
            
            notification_id = cursor.lastrowid
        
        return notification_id
    
    def get_notification_log(self, household_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get notification audit log for a household."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM notifications_log 
                WHERE household_id = ?
                ORDER BY sent_at DESC
                LIMIT ?
            """, (household_id, limit))
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    # ────────────── CHATBOT CONVERSATION MANAGEMENT ──────────────────────────────
    
    def save_conversation(self, household_id: str, email: str, question: str,
                         answer: str, language: str = "en",
                         tool_calls: Optional[List[str]] = None,
                         grounding_data: Optional[Dict[str, Any]] = None,
                         is_valid: bool = True) -> int:
        """
        Save a chatbot conversation turn.
        Returns conversation ID.
        """
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        grounding_data_json = json.dumps(grounding_data) if grounding_data else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chatbot_conversations
                (household_id, email, message_type, language, question, answer,
                 tool_calls_json, grounding_data_json, is_valid_grounded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (household_id, email, 'question', language, question, answer,
                 tool_calls_json, grounding_data_json, is_valid))
            
            conversation_id = cursor.lastrowid
        
        return conversation_id
    
    def get_conversation_history(self, household_id: str, email: str, 
                                limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversation history for a user in a household."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chatbot_conversations 
                WHERE household_id = ? AND email = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (household_id, email, limit))
            rows = cursor.fetchall()
        
        # Reverse to get chronological order
        return [dict(row) for row in reversed(rows)]

    def clear_conversations(self, household_id: str, email: str) -> int:
        """Delete chatbot history for a user in a household. Returns rows deleted."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM chatbot_conversations
                WHERE household_id = ? AND email = ?
            """, (household_id, email))
            return cursor.rowcount

    def get_household_settings(self, household_id: str) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM household_settings WHERE household_id = ?",
                (household_id,),
            )
            row = cursor.fetchone()
        if row:
            return dict(row)
        self.ensure_household_settings(household_id)
        return {
            "household_id": household_id,
            "bill_threshold": 1500.0,
            "last_bill_alert_at": None,
            "last_weekly_summary_at": None,
            "last_optimization_key": None,
            "onboarding_completed": False,
            "onboarding_completed_at": None,
        }

    def ensure_household_settings(self, household_id: str) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO household_settings (household_id)
                VALUES (?)
            """, (household_id,))

    def update_household_settings(self, household_id: str, updates: Dict[str, Any]) -> None:
        allowed = {
            "bill_threshold", "last_bill_alert_at",
            "last_weekly_summary_at", "last_optimization_key",
            "onboarding_completed", "onboarding_completed_at",
        }
        updates = {k: v for k, v in updates.items() if k in allowed}
        if not updates:
            return
        self.ensure_household_settings(household_id)
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [household_id]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE household_settings SET {set_clause} WHERE household_id = ?",
                values,
            )

    def is_onboarding_completed(self, household_id: str) -> bool:
        """Check if user has completed onboarding."""
        settings = self.get_household_settings(household_id)
        return bool(settings.get("onboarding_completed", False))

    def mark_onboarding_completed(self, household_id: str) -> None:
        """Mark onboarding as completed for a household."""
        self.update_household_settings(household_id, {
            "onboarding_completed": True,
            "onboarding_completed_at": datetime.now().isoformat(),
        })
    
    # ────────────── UTILITIES ──────────────────────────────────────────────────
    
    @staticmethod
    def _validate_email(email: str) -> bool:
        """Simple email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def clear_old_conversations(self, days: int = 30) -> int:
        """Clean up old conversation history. Returns number deleted."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM chatbot_conversations 
                WHERE created_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            deleted = cursor.rowcount
        
        return deleted


# Singleton instance
_db = None

def get_db() -> DatabaseManager:
    """Get or create the database manager singleton."""
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db
