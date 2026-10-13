# Deploying EnergyPulse

## Streamlit Community Cloud (free, simplest)

1. Put this folder in a GitHub repository (private is fine).
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**.
3. Pick the repository and branch, set **Main file path** to `app.py`, and under
   **Advanced settings** choose Python 3.11.
4. In **Advanced settings > Secrets**, add:

   ```
   ENERGYPULSE_SECRET = "a long random string you keep private"
   ```

   Add the SMTP settings from `.env.example` there too if you want email alerts.
   `ENERGYPULSE_ALLOW_PRIVATE_API` should stay unset on a public host.
5. Press **Deploy**. The first build takes several minutes (TensorFlow is large).

`requirements.txt` lists the Python packages and `packages.txt` installs Tesseract for
bill OCR; the host reads both automatically.

## Render, Railway or Heroku

The `Procfile` and `runtime.txt` are already in place. Create a web service from the
repository, set the `ENERGYPULSE_SECRET` environment variable, and deploy. Choose an
instance with at least 1 GB of memory.

## Know before you present

- **Storage is temporary on free hosts.** Accounts, saved bills and the notification
  history live in `energypulse.db` and `data/bills/`, which are wiped when the app
  restarts or goes to sleep. Sign up again before a demo, or attach a persistent disk.
- **Set `ENERGYPULSE_SECRET`.** Without it a new key is generated on each restart and
  earlier seals and encrypted bills can no longer be verified or opened.
- **Memory.** TensorFlow (for the LSTM) needs most of a 1 GB host. If the host stops the
  app for using too much memory, set `ENERGYPULSE_DISABLE_LSTM = "1"` in Secrets: the
  next-hour forecast then uses XGBoost alone, which is only slightly less accurate
  (see `models/forecast_meta.json`).
- **Light and dark mode** is a setting of the running server, so all visitors share it.
- **First visit after a restart is slow** while the models load.
