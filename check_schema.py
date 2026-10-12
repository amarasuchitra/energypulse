from db import DatabaseManager
db = DatabaseManager('data/energypulse.db')
print('DB initialized')
import sqlite3
conn = sqlite3.connect('data/energypulse.db')
c = conn.cursor()
c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'")
result = c.fetchone()
if result:
    print(result[0])
else:
    print("users table not found")
conn.close()