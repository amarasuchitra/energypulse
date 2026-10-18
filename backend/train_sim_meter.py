"""
Retrain only the simulated meter's next-hour model (about 6-10 minutes) and
update its entry in models/forecast_meta.json.  The real-home models are left
as they are.

    python backend/train_sim_meter.py && python backend/model_report.py
"""
import json
import os
import time

import model as M


def main():
    path = os.path.join(M.MODEL_DIR, "forecast_meta.json")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    started = time.time()
    M._log("== Simulated meter: next-hour model trained on simulated homes ==")
    meta["simulated_meter"] = M.train_simulated_meter(meta["xgb_params"])
    meta["simulated_meter"]["trained"] = time.strftime("%Y-%m-%d %H:%M")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    M._log(f"Saved in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
