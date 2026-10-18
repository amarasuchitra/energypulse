"""Make backend/ and frontend/ importable for the tests, as frontend/app.py does for the app."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in (os.path.join(ROOT, "frontend"), os.path.join(ROOT, "backend")):
    if d not in sys.path:
        sys.path.insert(0, d)
os.chdir(ROOT)          # a few tests read data/ and models/ by relative path
