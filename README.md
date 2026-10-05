# ⚡ EnergyPulse — Home Energy Dashboard

A **Streamlit** web app for household electricity forecasting, appliance
breakdown, bill forecasting and anomaly tips, in **English, Hindi, Kannada and
Telugu**.

It ships with the **UCI Individual household electric power consumption**
dataset (34,168 hourly rows, December 2006 – November 2010) and a trained
**XGBoost + LSTM hybrid**, so every tab works out of the box.

> The bundled data is a real meter recording from one household, used as a
> **demo sample**. The dashboard labels it as sample data, shifts its timestamps
> forward so it lines up with the present, and never presents it as a live
> meter feed. To use your own readings, upload a CSV — see
> [Using your own data](#using-your-own-data).

## Weather, household habits and left-on appliances (V4)

- **Weather.** `weather.py` simulates the daily high temperature for the household's city (seasonal curve plus warm and cool spells; the same city and date always give the same value). The simulator runs the air conditioner longer on hotter days and not at all on cool ones. The daily forecast fits AC use to the degrees above 24 C today and yesterday. Measured on the last 14 days of several simulated homes, the forecast error fell from 18-27% to 12-18%.
- **Home size and city** now change the simulated home (background load, cooling time, climate), along with usage levels and number of people.
- **Left-on appliances.** The detection model is retrained with days where appliances are left on for hours, and with homes of varied habits and sizes. On-off F1 on unseen simulated homes: fridge 0.98, AC 0.99, geyser 0.99, washing machine 0.92, water pump 0.96, microwave 0.94; on unseen left-on days: 0.97, 0.98, 0.98, 0.88, 0.86, 0.72. Known weak case: a microwave left on for a long time underneath a running geyser is usually missed.

All figures are from simulated homes and simulated weather; real homes will score lower.

## Household toolkit (V3.9)

Pages that answer the other questions a household has about electricity. The sums are in `toolkit.py` (tested in `test_toolkit.py`), the pages in `toolkit_ui.py`.

| Page | What it does |
|---|---|
| My Bills | Works a bill out again from its units, slabs, fixed charge and duty and compares it with the amount charged; warns before a slab is crossed; explains a change by appliance |
| Safety | Highest sustained load against the sanctioned load, a fault history, and left-on notifications |
| Upgrades | Payback of a 5-star replacement, rooftop solar with subsidy and net metering, inverter and battery size |
| Goals | Monthly rupee goal with streak, planned hours per appliance, simulated similar homes, carbon |
| Family | Who switched what, and a fingerprinted one-page weekly PDF |
| Notifications | Share the daily forecast on WhatsApp; SMS to members with a phone number when Twilio is set up |
| Help | Power-cut log and a complaint letter filled with the household's own figures |

Tariff tables for Karnataka and Delhi are starting points and every rate is editable, because tariff orders change each year. Replacement savings, solar yield and prices are stated assumptions the user can change. The app advises only; without hardware it cannot switch anything.

## One flow from sign-up to savings (V3.8)

The app now runs as one connected path:

1. **Sign up with a strong password.** At least 10 characters with upper and lower case, a number and a symbol; common passwords and ones containing the email name are refused, and the password is typed twice (`auth.py`).
2. **List your appliances once.** The setup is saved per household, so signing in again goes straight to your home (`db.py`, table `household_homes`).
3. **The 3D home shows exactly what you listed, and nothing else.** An item you did not add is not drawn; a second unit of the same type is shown from its switch. The six appliances the detection model knows are found from the meter; anything else (TV, fans, lights, iron, custom items) is drawn in a room and shown from its switch (`devices.py`).
4. **Every device has a switch and a description.** One list on the Home page; the refrigerator's switch is locked on.
5. **A daily prediction is sent as a notification.** Expected units and cost for today by appliance, the measured error of the forecast on the last 14 days, and the changes worth making with the arithmetic behind each rupee figure (`daily_brief.py`). It is shown on Home and Overview, kept on the Notifications page, and emailed when email is configured.

The forecast is created the first time the app is opened each day; a free host cannot send it while the app is asleep.

## Motion and feedback (V3.7)

`motion.py` holds the whole motion system so every page behaves the same way.

- **Timing:** three durations (120, 200, 380 ms; 600 ms for entrances) and one easing curve.
- **Arrival:** pages fade and rise on navigation; cards and sections rise in as they scroll into view, staggered.
- **Numbers and charts:** figures count up on first sight and ease to new values; charts draw in and ease between values.
- **Feedback:** toasts in four kinds (success, warning, error, information) via `notify(kind, text)`; inline messages slide in, errors with a small shake.
- **Loading:** skeleton shapes while the meter history is prepared the first time, and inside the Home console until the 3D view is ready.
- **Controls:** hover, press and focus states on buttons, cards, inputs, selects, sliders and switches; a blurred backdrop behind dialogs.
- **Restraint:** everything is off under `prefers-reduced-motion`, and shortened on low-powered devices.

## Accounts, data protection and tamper checks (V3.4)

| File | What it does |
| --- | --- |
| `auth.py` | Accounts: salted PBKDF2 password hashes, one registration per email, lockout after five wrong passwords, 8-hour sessions, a private household for every guest, encryption of stored bill files |
| `integrity.py` | Tamper evidence: every saved bill is fingerprinted, chained to the record before it and sealed; exported reports carry a seal that breaks if they are edited |

- Every page reads data by the household of the signed-in account, never by anything typed into a form.
- Keys and household data are kept out of git (`data/.integrity_key`, `data/bills/`, `energypulse.db`).
  Set `ENERGYPULSE_SECRET` in production, and `ENERGYPULSE_METER_TOKEN` to require a token from the meter.
- Limits: a bill forged before upload cannot be detected; someone holding both the database and the key
  could rewrite records; family members do not have their own sign-in.

The Home console also has ceiling fans you can switch, hover cards for every device, a full 360-degree
view, light and dark mode (toggle in the sidebar), and a household activity feed that records who
switched what.

## Interface (V3.3)

The site is organised as pages reached from a left navigation, not a tab bar:
Home, Overview, Appliances, Analysis, Save Energy, My Bills, Trends, Family,
Notifications, Energy Chat and Settings. Only the page on screen is computed.
Controls that used to fill the sidebar (tariff, peak hours, demo home, data
upload, language, sign out) are on the Settings page.

`shell_ui.py` holds the whole look: the stylesheet, navigation, page headers and
the sign-in panel. Type uses three faces, bundled in `static/fonts/` so the app
works offline: Sora for titles, Hanken Grotesk for interface text,
IBM Plex Mono for readings. No emoji is used anywhere; `i18n.py` strips them from
every translated string in one place.

## Home console and the simulated meter (V3.1)

The **Home** tab is a 3D model of the home driven by the main-meter reading:
appliances the detection model finds light up, the supply route from the meter
to each one is traced on the floor, and a side panel shows the meter (kW now,
units today, cost, current tariff period) and every device's state and cost.

**The meter is a simulation; no hardware is involved.** `meter_sim.py` writes a
minute-by-minute power pattern for each appliance the household listed, adds
them up with a background load and noise, and that sum is the meter reading.
The detection model is given only the sum. **Test** mode turns the user's own
switches into a new reading. A device switched on stays on until the user
switches it off and confirms it. The page carries a "How the meter works" note.

The simulated home follows the household: the usage level chosen for each
appliance and the number of people change how long and how often things run.
The Home page and the Overview sit at the current time on the household's
clock and show only what has happened so far today. The forecast model trained
on the recorded reference dataset is shown separately on the Trends page.

| File | What it does |
| --- | --- |
| `meter_sim.py` | The simulated home and meter |
| `home_ui.py` | Prepares the day's data and hosts the console |
| `home_component/` | The console itself (HTML, CSS, JavaScript, three.js bundled locally, works offline) |
| `clock.py` | The household's time zone, confirmed during setup |

## Appliance detection from the main meter (V3)

The **Appliances** tab works from one signal only: total power at the main meter,
per minute. A detection model (`disaggregate.py`) estimates which appliances are
on and how much each is drawing. From that the app reports the biggest consumers,
how long each appliance ran, abnormally long runs, and realistic suggestions.

| File | What it does |
| --- | --- |
| `appliance_profiles.py` | Appliance categories (always-on, shiftable, comfort, on-demand) and what advice each may receive |
| `meter_sim.py` | Simulated 1-minute main-meter signal, used for training and for test mode |
| `disaggregate.py` | Detection model plus a rule-based baseline, with scoring |
| `sessions.py` | Usage sessions, run durations, long-run flags, fridge duty cycle |
| `tariff.py` | Flat or time-of-day tariff |
| `scheduler.py` | Category-aware recommendations with shown calculations |
| `appliance_ui.py` | The Appliances tab and test mode |

> **The meter signal in this tab is simulated.** Accuracy in `models/nilm_metrics.json`
> is measured on simulated homes the model did not train on, and will be lower on a
> real home. To train on a real recording, use `disaggregate.load_labelled_csv()`.

Retrain with `python disaggregate.py`; run the tests with
`pytest test_appliance_intelligence.py`.

## New Features (V2.1)

**Three new fully-implemented features** have been added to EnergyPulse:

### 1. Family Member Management
- Add household members with individual notification preferences
- Each member can choose to receive bill alerts and/or optimization tips
- Personalized language preference per member
- Immediate removal stops all notifications (fully enforced, not just hidden)
- Validation: reject invalid emails and duplicates with language-aware error messages
- Linked to primary user via household ID (not separate, disconnected records)

### 2. Automated Notifications (Email & SMS)
- **Bill Alerts**: Triggered by predicted cost threshold or schedule, content uses only real computed numbers
- **Optimization Tips**: Triggered by real usage anomalies detected in data, each tip references specific real appliance and measured increase
- Email via Gmail SMTP, SendGrid, or test mode
- SMS via Twilio (optional)
- Each recipient in their preferred language
- Full audit trail in notification log (who, when, what real data triggered it, status)
- Graceful failure handling: errors logged, non-blocking status shown, app never crashes
- Test notification button to verify email delivery without waiting for real trigger
- "Run scheduled checks now" button triggers detection immediately from real data

#### Email configuration — Gmail SMTP (recommended, no signup)

Copy the template first:

```
copy .env.example .env        # Windows PowerShell
cp .env.example .env          # macOS / Linux
```

The app only needs **two** variables. Everything else has a working default,
so a shorter `.env` is a valid `.env`:

| Variable | Required | Default when unset | Notes |
| --- | --- | --- | --- |
| `SMTP_USER` | **yes** | — | Your Gmail address. `SMTP_USERNAME` is still accepted as an older alias. |
| `SMTP_PASSWORD` | **yes** | — | The 16-character **app password**, not your normal Gmail password. |
| `SMTP_HOST` | no | `smtp.gmail.com` | |
| `SMTP_PORT` | no | `587` | Port `465` is detected as implicit TLS automatically. |
| `SMTP_FROM` | no | same as `SMTP_USER` | Set only to send from a different address. |
| `SMTP_STARTTLS` | no | `true` | Only disable if your provider says to. |
| `SMTP_USE_SSL` | no | `false` | Implied by port `465`. |

Get a free Gmail app password in about three minutes (no third-party accounts):

1. **Turn on 2-Step Verification** — Google Account → Security → *2-Step Verification* (required once for app passwords).
2. **Create an app password** — Google Account → Security → *App Passwords* → **Mail** (or Other → name it `energypulse`).
3. **Copy the 16-character password** it shows (e.g. `abcd efgh ijkl mnop`).
4. **Put those two values in `.env`** in the project root:
   ```
   SMTP_USER=your@gmail.com
   SMTP_PASSWORD=the-16-character-app-password
   ```
5. **Save and restart the app.** The Notifications tab switches to **Configured** and, if anything is still absent, names the exact variables it is waiting for. Press **Send test** to verify delivery — the tab only reports "Connected" after a message has actually gone out, and a failure shows the mail server's own error text.

Values are read with `python-dotenv` and fall back to `.streamlit/secrets.toml`
(flat keys or a `[notifications]` section both work), so you can keep
credentials out of the working tree entirely. The environment wins when both
are set. Neither `.env` nor `secrets.toml` is tracked by git.

#### Email configuration — SendGrid (alternative)

If you prefer SendGrid instead of Gmail, create a free account at sendgrid.com, then in `.env` set the variable the app reads:

```
SENDGRID_API_KEY=your_sendgrid_api_key_here
```

Use **either** the SMTP block **or** `SENDGRID_API_KEY`, not both — the app prefers SendGrid when both are present. If no email variables are set at all, the app runs honestly: no email is sent, delivery status is recorded as failed, and the tab shows "Not configured" together with the list of variables it still needs.

A mistyped `SMTP_PORT` (anything that is not a number between 1 and 65535) is
reported as a warning and falls back to port 587. It does not stop the app.

#### SMS — optional (future step)

SMS uses Twilio (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`). It is intentionally optional — leave the variables blank and SMS reads **"Optional — not configured"** rather than an error. It is not required for email delivery, and installing the `twilio` package is optional too.

#### Verifying the setup

- The two metrics on the Notifications tab report the **actual** state: `Not configured` (with the missing variable names), `Configured`, `Connected` (a real send succeeded) or `Last send failed` (with the provider's message).
- Every attempt, successful or not, is written to **Notification history** with its status and error text, so a failure can always be matched to the click that caused it.
- Credentials are never printed, logged, or written to the database. If a provider echoes a secret inside its own error string, it is masked before display.

### 3. Grounded Q&A Chatbot
- Natural language questions about household energy usage
- **Answers ONLY from real household data** via tool calls:
  - `get_usage_history`: Actual measurements
  - `get_appliance_breakdown`: Real appliance-level breakdown
  - `get_current_prediction`: Computed from recent data
  - `get_week_comparison`: This week vs. previous week (only when a real prior week exists)
  - `get_optimization_tips`: Detected anomalies, not generic suggestions
  - `get_family_notification_log`: Actual sent notifications for the household
- Multi-language support: questions understood and answered in all 4 languages
- **No hallucinations**: If data is unavailable (or there is no usable prior week), the chatbot says so honestly
- Questions that match nothing in scope (weather, sports, general trivia, an empty box) are refused in the user's own language rather than answered with household figures
- All numeric claims validated against tool output using LLMHallucinationGuard (including date/time tokens, which are excluded from number matching)
- Conversation history stored per user, clearable from the chat tab

See [FEATURES_NEW.md](FEATURES_NEW.md) for complete documentation.

## Pages

After signing in (or continuing as guest) the app shows a dashboard with eight tabs:

| Tab | What it does |
| --- | --- |
| 🏠 **Overview** | Project intro, headline metrics, pipeline explainer |
| 📊 **Analysis** | Interactive time series, appliance breakdown, hour×weekday heatmap, weekday profile, actual-vs-predicted overlay |
| 💡 **Save** | Optimization suggestions and cost reduction opportunities |
| 🧾 **Bills** | Cost details, tariff settings |
| 📈 **Trends** | 24-hour forecast with confidence band, recent trends |
| 👨‍👩‍👧 **Family** | Manage household members and their notification preferences |
| 📨 **Notifications** | Backend status, alert thresholds, run checks now, test emails, audit log |
| 💬 **Energy Chat** | Grounded multilingual Q&A about your real usage |

## Getting started

Requires Python 3.10+.

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the dashboard
streamlit run app.py
```

The app then opens at `http://localhost:8501`.

## Notification setup

Email and SMS are **off until you supply credentials**, and the Notifications tab
says so honestly instead of pretending to be connected. Nothing else in the app
depends on them — the dashboard runs fully without this step.

**1. Create the template**

```bash
copy .env.example .env        # Windows PowerShell
cp .env.example .env          # macOS / Linux
```

**2. Fill in email (required for alerts)**

The only two values you must set are the account and a Gmail app password.
`SMTP_HOST`, `SMTP_PORT`, TLS and `SMTP_FROM` all have working defaults, so a
two-line `.env` is valid:

```ini
SMTP_USER=you@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop
```

To get the app password: Google Account → Security → 2-Step Verification (on) →
Security → App Passwords → **Mail**. Use the 16-character app password, not your
normal Google password — Google rejects the normal one for SMTP.

<details>
<summary>Full variable reference (all optional unless marked)</summary>

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `SMTP_USER` | **yes** | — | Sending account. `SMTP_USERNAME` works as an alias. |
| `SMTP_PASSWORD` | **yes** | — | App password for `SMTP_USER`. |
| `SMTP_HOST` | no | `smtp.gmail.com` | SMTP server. |
| `SMTP_PORT` | no | `587` | `587` = STARTTLS, `465` = implicit TLS (auto-detected). |
| `SMTP_FROM` | no | `SMTP_USER` | Different sender address. |
| `SMTP_STARTTLS` | no | `true` | Leave on unless your provider says otherwise. |
| `SMTP_USE_SSL` | no | `false` | Implied by port `465`. |
| `SENDGRID_API_KEY` | no | — | SendGrid instead of SMTP; wins if both are set. |

</details>

Values load via `python-dotenv`, falling back to `.streamlit/secrets.toml` if the
environment variable is absent. Both flat keys and a `[notifications]` section
are understood:

```toml
# .streamlit/secrets.toml
SMTP_USER = "you@gmail.com"
SMTP_PASSWORD = "abcd efgh ijkl mnop"
```

`.env` and `.streamlit/secrets.toml` are both in `.gitignore`. Credentials are
never printed, logged, or stored — including when a provider echoes a secret
back inside its own error message, which is masked before display.

**3. SMS is optional**

Leave `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `TWILIO_FROM_NUMBER` unset
and SMS reads **"Optional — not configured"** (not an error), with alerts going
out by email only. The `twilio` package is optional too.

**4. Restart and verify**

Restart the app after editing `.env` — configuration is read once at startup.
The tab then shows **Configured**, or names the variables still missing. Press
**Send test** to deliver a real message to the selected recipient; success and
failure messages show the mail server's own text, and every attempt is recorded
in **Notification history** with its status and error.

## Project structure

```
majorproject/
├── app.py                  # entry point (auth, dashboard + 8 tabs)
├── chatbot.py              # grounded multilingual chatbot (tool-calling pipeline)
├── features_ui.py          # Family / Notifications / Chat tab renderers
├── notifications.py        # email (SendGrid/SMTP) + SMS (Twilio) dispatch
├── llm_guard.py            # hallucination guard for numeric claims
├── i18n.py                 # English/Hindi/Kannada/Telugu translations
├── db.py                   # SQLite: users, family, notifications, chat
├── cost.py                 # tariff/weekly/peak cost helpers
├── data.py                 # UCI download, hourly aggregation, loaders
├── data_source.py          # upload normalisation, units, capability detection
├── appliances.py           # NILM sub-meter → appliance breakdown
├── optimize.py             # optimization/insight rules
├── model.py                # hybrid training, metrics, forecasting
├── replay.py               # replays the active frame row by row
├── convert_to_docx.py      # exports the conference paper to .docx
├── test_csv_import.py      # 26 tests: upload formats, units, timestamps
├── test_chatbot_routing.py # 38 tests: intent routing + guard number checks
├── test_notifications.py   # 12 tests: backend status, ports, log types
├── test_new_features.py    # feature suite (6 test groups, all green)
├── requirements.txt
├── .env.example            # email/SMS credential template
└── .streamlit/config.toml  # dark theme
```

## How the model works

- **Data**: the UCI household dataset, 1-minute readings aggregated to
  **34,168 hourly rows**. The three `Sub_metering_*` columns are reported by the
  source in **Wh per minute**; they are multiplied by 60 to get hourly Wh, which
  is why the appliance breakdown no longer collapses into a single "Other" bar.
- **Model inputs (10)**: the last 10 hours (`WINDOW_SIZE`) of
  `Global_active_power`, `hour`, `day_of_week`, `is_weekend`,
  `Global_reactive_power`, `Voltage`, `Global_intensity` and the three
  sub-meters.
- **Model**: an **XGBoost regressor blended with an LSTM**, weighted by
  inverse MAE. Trained time-ordered on the first 80% of hours, evaluated on the
  most recent 20%. Current metrics from `models/model_meta.pkl`:
  - XGBoost MAE **0.0120**, LSTM MAE **0.3399**
  - hybrid MAE **0.0163** at weights **0.966 / 0.034** (a flat 50/50 blend
    scores 0.1697, roughly ten times worse)
- **Honesty guards**: prediction is withheld unless every one of the 10 input
  columns is present; appliance breakdown and optimization need all three
  sub-meters. Missing inputs produce a localized, actionable message instead of
  a number.
- The LSTM is optional at inference. If it is unavailable or the dataset shape
  is unsupported, the app falls back to XGBoost and records the reason.

## Using your own data

Upload a CSV from the **Data** panel. The uploader accepts `.csv` and `.tsv`,
sniffs the delimiter, and reads Latin-1 as well as UTF-8.

- **Timestamps** may be a single `Date`+`Time` pair or one combined column.
  ISO dates are read unambiguously; for `dd/mm` vs `mm/dd` the order is
  inferred per column, and an ambiguous month-first reading is flagged in the
  provenance banner rather than silently reinterpreted.
- **Units** are detected per column. Sub-hourly data is resampled to hourly:
  `Wh` values are summed and `W` values averaged, with a warning and a manual
  override when the unit is not certain.
- **Column names** are matched through an alias table, so `total_kwh`,
  `Total Consumption` and `Global_active_power` all map to the same field.
- **Timestamps are preserved.** The dashboard never rewrites an upload's dates;
  the current-time match used for the sample does not apply to uploads, whose
  headline reading is simply the newest row in the file.

Anything the upload cannot supply is withheld with a message naming the missing
columns, rather than filled with a default or estimated number. The panel always
shows the file name, row count, time range, and units actually applied.

## Bill photo OCR (optional)

Uploading a bill **image** runs OCR to prefill units, amount and billing period.
This needs the Tesseract engine, which is a separate program from the Python
packages:

- Windows: `winget install UB-Mannheim.TesseractOCR`
- macOS: `brew install tesseract`
- Debian/Ubuntu: `sudo apt install tesseract-ocr`

Without the engine the app says OCR is unavailable and lets you fill the fields
in manually. It never guesses at the contents of an image it could not read.

## Costs & emissions assumptions

- Default price: **₹8 / kWh** (configurable per household in the app).
- Emission factor: **0.386 kg CO₂ / kWh** (typical grid mix).
