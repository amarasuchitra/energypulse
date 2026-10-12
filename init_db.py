from db import DatabaseManager
db = DatabaseManager('data/energypulse.db')
print('DB initialized')
import sqlite3
conn = sqlite3.connect('data/energypulse.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in c.fetchall():
    print(row[0])
conn.close()