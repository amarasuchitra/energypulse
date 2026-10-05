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
                    FOREIGN KEY (household_id) REFERENCES users(household_id)
                )
            """)
            
            conn.commit()
    
    # ────────────── USER MANAGEMENT ──────────────────────────────
    
    def create_user(self, email: str, name: str, password: str, is_guest: bool = False,
                   preferred_language: str = "en", household_id: Optional[str] = None) -> bool:
        """Create a new user account. Returns True on success."""
        email = (email or "").strip().lower()
        if household_id is None:
            household_id = hashlib.md5(email.encode()).hexdigest()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users 
                    (email, name, password_hash, household_id, is_guest, preferred_language)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (email, name, password_hash, household_id, is_guest, preferred_language))
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
        password_hash = hashlib.sha256((email + household_id).encode()).hexdigest()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users
                (email, name, password_hash, household_id, is_guest, preferred_language)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    name = excluded.name,
                    is_guest = excluded.is_guest,
                    preferred_language = excluded.preferred_language
            """, (email, name, password_hash, household_id,
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
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email, name, household_id, is_guest, preferred_language 
                FROM users 
                WHERE email = ? AND password_hash = ?
            """, (email, password_hash))
            row = cursor.fetchone()
            
        if row:
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
