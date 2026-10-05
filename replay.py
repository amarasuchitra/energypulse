"""
PART 6 — Simulated Real-Time Replay
=====================================
Since there is no physical smart meter, this module replays
cleaned_energy_data.csv row-by-row with a configurable delay
to simulate a live data stream.

The dashboard (app.py) creates a ReplaySimulator instance and stores it
in st.session_state so it persists across Streamlit reruns.  The background
thread keeps advancing through the data while the dashboard reads the
"latest row" to display live metrics.

Usage in Streamlit:
    from replay import ReplaySimulator
    sim = ReplaySimulator(delay=3.0)
    sim.start()
    row = sim.latest_row        # dict with the current row
    window = sim.get_window(50) # last 50 rows as a DataFrame

Standalone test:
    python replay.py
"""

import os
import time
import threading

import pandas as pd

# Path to the cleaned CSV produced by data.py
CLEAN_CSV = os.path.join("data", "cleaned_energy_data.csv")


class ReplaySimulator:
    """
    Replays historical electricity data one row at a time in a background
    thread, mimicking a live smart-meter feed.
    """

    def __init__(self, delay: float = 3.0, csv_path: str = CLEAN_CSV):
        """
        Parameters
        ----------
        delay : float
            Seconds to wait between consecutive rows.
            Default is 3 seconds (one row = one simulated "hour").
        csv_path : str
            Path to the cleaned CSV file.
        """
        self.delay = delay
        self.csv_path = csv_path
        self._df: pd.DataFrame = pd.DataFrame()
        self._index: int = 0
        self._current_row: dict | None = None
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._load_data()

    def _load_data(self):
        """Load the cleaned CSV into memory."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"{self.csv_path} not found.\nRun:  python data.py"
            )
        self._df = pd.read_csv(self.csv_path, parse_dates=["datetime"])

    def _replay_loop(self):
        """Background loop: advance one row every `self.delay` seconds."""
        while self._running and self._index < len(self._df):
            with self._lock:
                row = self._df.iloc[self._index]
                self._current_row = row.to_dict()
                self._index += 1
            time.sleep(self.delay)

    # ── Public API ──────────────────────────────────────────────────

    def start(self):
        """Start the background replay thread. Safe to call multiple times."""
        if self._running:
            return
        self._running = True
        self._index = 0
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Pause the replay (can be resumed with start())."""
        self._running = False

    def reset(self):
        """Stop and rewind to the beginning."""
        self.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.delay + 0.5)
        with self._lock:
            self._index = 0
            self._current_row = None

    def set_data(self, df: pd.DataFrame, restart: bool = True) -> None:
        """
        Replay a different frame (e.g. a user upload) instead of the bundled CSV.

        This rewinds the replay so the newly supplied data is the only thing
        ever surfaced.
        """
        if df is None or df.empty:
            return
        was_running = self._running
        self.stop()
        with self._lock:
            self._df = df.reset_index(drop=True)
            self._index = 0
            self._current_row = None
        if restart or was_running:
            self.start()

    @property
    def latest_row(self) -> dict | None:
        """Return the most recently replayed row as a dict."""
        with self._lock:
            return self._current_row

    @property
    def progress(self) -> tuple:
        """Return (rows_played_so_far, total_rows_in_dataset)."""
        return self._index, len(self._df)

    @property
    def is_finished(self) -> bool:
        return self._index >= len(self._df)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_window(self, n: int = 50) -> pd.DataFrame:
        """Return the last `n` rows replayed so far."""
        with self._lock:
            if self._index == 0:
                return pd.DataFrame()
            start = max(0, self._index - n)
            return self._df.iloc[start : self._index].copy()

    def get_full_history(self) -> pd.DataFrame:
        """Return ALL rows replayed so far."""
        with self._lock:
            if self._index == 0:
                return pd.DataFrame()
            return self._df.iloc[: self._index].copy()

    def get_live_row_for_current_time(self, full_df: pd.DataFrame) -> dict | None:
        """
        Return a row from full_df whose hour-of-day matches the current
        real-world time.

        This is a demo affordance for the SAMPLE dataset, whose timestamps were
        shifted forward: if it is 3 PM now, the row originally recorded around
        3 PM carries today's date, so the headline tracks the clock.

        It must NOT be used for a user upload. Those timestamps are real, so a
        row matching the current hour-of-day could be from any day in the file
        and would be a fabricated "current" reading. Callers are expected to
        use :attr:`latest_row` there instead.

        Parameters
        ----------
        full_df : pd.DataFrame
            The full dataset loaded by the dashboard.

        Returns
        -------
        dict or None - the matching row as a dict, or None if not found.
        """
        now = pd.Timestamp.now()
        current_hour = now.hour

        if full_df is None or full_df.empty:
            return None

        if "datetime" not in full_df.columns:
            return full_df.iloc[-1].to_dict()

        # Only treat a row as "current" when its full timestamp is recent.
        recent_cutoff = now - pd.Timedelta(hours=2)
        fresh = full_df[full_df["datetime"] >= recent_cutoff]
        if not fresh.empty:
            return fresh.iloc[-1].to_dict()

        # Find rows with matching hour-of-day, return the latest one
        hour_mask = full_df["datetime"].dt.hour == current_hour
        matches = full_df[hour_mask]

        if matches.empty:
            # Fall back to the most recent row in the dataset
            return full_df.iloc[-1].to_dict()

        return matches.iloc[-1].to_dict()


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Replay Simulator — standalone test")
    print("Press Ctrl+C to stop.\n")

    sim = ReplaySimulator(delay=1.0)
    sim.start()

    try:
        while not sim.is_finished:
            row = sim.latest_row
            if row:
                ts = row.get("datetime", "?")
                gap = row.get("Global_active_power", 0)
                idx, total = sim.progress
                print(f"  [{idx}/{total}] {ts}  —  {gap:.3f} kW")
            time.sleep(0.5)
    except KeyboardInterrupt:
        sim.stop()
        print("\nStopped by user.")

    print("Replay finished.")
