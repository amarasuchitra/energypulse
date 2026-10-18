"""
Where the project's files live, as absolute paths from the project root, so the
app, the scripts and the tests work whichever folder they are started from.

    energypulse/
    ├── frontend/   screens (Streamlit), 3D home, styles
    ├── backend/    forecasting, optimiser, alerts, database, auth  (this file)
    ├── models/     trained models
    ├── data/       dataset, bills, photos
    └── docs/       reports
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
DOCS_DIR = os.path.join(ROOT, "docs")
DB_PATH = os.path.join(ROOT, "energypulse.db")
