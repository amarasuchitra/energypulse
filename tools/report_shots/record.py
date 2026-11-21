"""Record a screen video of the app running on localhost (CI only), following the demo script.

Every step writes a "keep" window to clips.json so the waits for loading can be cut afterwards.
"""
import json
import os
import time
import traceback

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = "video"
EMAIL, PW = os.environ["SHOT_EMAIL"], os.environ["SHOT_PW"]
os.makedirs(OUT, exist_ok=True)
SIZE = {"width": 1440, "height": 900}
CLIPS = []          # (video name, label, start s, end s)

FIND_SCROLLER = """() => {
  let best = null, bestH = 0;
  for (const e of document.querySelectorAll('*')) {
    const st = getComputedStyle(e);
    if (!/(auto|scroll)/.test(st.overflowY)) continue;
    const extra = e.scrollHeight - e.clientHeight;
    if (extra > 50 && e.clientHeight > 300 && e.scrollHeight > bestH) { best = e; bestH = e.scrollHeight; }
  }
  window.__epScroller = best || document.scrollingElement;
  return window.__epScroller.scrollHeight - window.__epScroller.clientHeight;
}"""


class Cam:
    def __init__(self, page, name):
        self.page, self.name, self.t0 = page, name, time.monotonic()

    def now(self):
        return time.monotonic() - self.t0

    def keep(self, label, fn):
        a = self.now()
        try:
            fn()
        except Exception:
            print("FAILED", label)
            traceback.print_exc()
        CLIPS.append((self.name, label, round(a, 2), round(self.now(), 2)))
        print(f"{self.name} {label}: {a:.1f}-{self.now():.1f}")


def settle(page, wait=4.0):
    time.sleep(wait)
    for _ in range(120):
        if not page.locator('[data-testid="stSpinner"], [data-testid="stSkeleton"]').count():
            break
        time.sleep(1)
    time.sleep(1.5)


def hold(s):
    time.sleep(s)


def top(page):
    page.evaluate(FIND_SCROLLER)
    page.evaluate("() => window.__epScroller.scrollTo(0, 0)")
    time.sleep(0.8)


def glide(page, px=None, speed=260, pause_every=650, pause=1.6):
    """Scroll down smoothly like a person reading: px pixels (or to the end) at `speed` px/s."""
    room = page.evaluate(FIND_SCROLLER)
    target = min(px, room) if px else room
    done, since = 0, 0
    step = speed / 30
    while done < target:
        page.evaluate("(d) => window.__epScroller.scrollBy(0, d)", step)
        done += step
        since += step
        time.sleep(1 / 30)
        if since >= pause_every:
            since = 0
            time.sleep(pause)
    time.sleep(1.5)


def click_key(page, key, wait=4.0):
    page.locator(f".st-key-{key} button").first.click()
    settle(page, wait)


def tab(page, name, wait=4.0):
    page.get_by_role("tab", name=name).first.click()
    settle(page, wait)


def slow_type(loc, text):
    loc.click()
    loc.press_sequentially(text, delay=90)


def light_mode(page):
    try:
        page.get_by_role("button", name="Light mode").first.click()
        settle(page, 8)
    except Exception:
        pass


with sync_playwright() as p:
    browser = p.chromium.launch()

    def ctx_for(name):
        return browser.new_context(viewport=SIZE, record_video_dir=f"{OUT}/{name}", record_video_size=SIZE,
                                   timezone_id="Asia/Kolkata", color_scheme="light")

    # ---------------------------------------------------------- 1. sign-in page and typing the account
    ca = ctx_for("a_signin")
    pa = ca.new_page()
    cam = Cam(pa, "a_signin")
    pa.goto(URL)
    pa.wait_for_selector(".st-key-login_email input", timeout=240_000)
    settle(pa, 3)
    cam.keep("sign-in page", lambda: hold(5))

    def sign_in():
        slow_type(pa.locator(".st-key-login_email input"), EMAIL)
        hold(0.6)
        slow_type(pa.locator(".st-key-login_pw input"), PW)
        hold(1.0)
        pa.get_by_role("button", name="Sign In").first.click()
        hold(2.5)
    cam.keep("typing and sign in", sign_in)
    pa.close()
    ca.close()

    # ---------------------------------------------------------- 2. home setup (guest, no account needed)
    cg = ctx_for("b_setup")
    g = cg.new_page()
    cam = Cam(g, "b_setup")

    def setup():
        g.goto(URL)
        g.wait_for_selector(".st-key-guest_btn button", timeout=120_000)
        click_key(g, "guest_btn", 6)
        light_mode(g)
    setup()

    def setup_show():
        hold(4)
        glide(g, 900)
        top(g)
        click_key(g, "ob_next1", 5)
        hold(3)
        g.get_by_role("tab", name="Kitchen").first.click()
        hold(4)
        glide(g, 600)
    cam.keep("home setup", setup_show)
    g.close()
    cg.close()

    # ---------------------------------------------------------- 3. the signed-in app
    cm = ctx_for("c_app")
    page = cm.new_page()
    cam = Cam(page, "c_app")
    page.goto(URL)
    page.wait_for_selector(".st-key-login_email input", timeout=240_000)
    settle(page, 2)
    page.locator(".st-key-login_email input").fill(EMAIL)
    page.locator(".st-key-login_pw input").fill(PW)
    page.get_by_role("button", name="Sign In").first.click()
    page.wait_for_selector(".st-key-dash_edit_appliances", timeout=300_000)
    settle(page, 25)
    light_mode(page)
    settle(page, 12)

    cam.keep("dashboard tiles", lambda: hold(7))
    cam.keep("dashboard scroll", lambda: (glide(page, speed=240), hold(2), top(page)))

    def edit():
        click_key(page, "dash_edit_appliances", 6)
        hold(4)
        glide(page, 900)
        hold(2)
    cam.keep("edit appliances", edit)
    try:
        click_key(page, "ap_cancel", 8)
    except Exception:
        traceback.print_exc()

    click_key(page, "nav_appliances", 10)
    cam.keep("appliances devices", lambda: (hold(4), glide(page, 1400), top(page)))
    tab(page, "What Meter Shows", 8)
    cam.keep("what meter shows", lambda: (hold(4), glide(page, 1600), top(page)))
    tab(page, "Where Power Goes", 8)
    cam.keep("where power goes", lambda: (hold(4), glide(page, 1600), top(page)))

    tab(page, "My Devices", 4)

    def photo_upload():
        up = page.locator('[class*="st-key-am_drop_zone"] input[type=file]').first
        up.scroll_into_view_if_needed()
        hold(2)
        up.set_input_files(os.path.join("tests", "fixtures", "ceiling_fan.jpg"))
        hold(3)
    cam.keep("photo upload", photo_upload)
    settle(page, 30)
    cam.keep("photo result", lambda: (hold(3), glide(page, 1200)))

    click_key(page, "nav_optimize", 10)
    cam.keep("optimize", lambda: (hold(4), glide(page, 2600), top(page)))
    tab(page, "Monthly Goals", 6)
    cam.keep("goals", lambda: (hold(4), glide(page, 900)))
    tab(page, "Upgrade Calculator", 6)
    cam.keep("upgrade calculator", lambda: (hold(4), glide(page, 700)))

    click_key(page, "nav_bills_safety", 10)
    cam.keep("bills", lambda: (hold(4), glide(page, 1800), top(page)))
    tab(page, "Safety", 6)
    cam.keep("safety", lambda: (hold(4), glide(page, 1200), top(page)))
    tab(page, "Alerts", 6)
    cam.keep("alerts", lambda: (hold(4), glide(page, 2400), top(page)))

    click_key(page, "nav_household", 8)
    cam.keep("household", lambda: hold(4))
    tab(page, "Energy Assistant", 5)

    def ask():
        hold(2)
        slow_type(page.locator(".st-key-chat_input input"), "How much will next month cost?")
        hold(1)
        page.get_by_role("button", name="Ask").first.click()
        hold(2)
    cam.keep("assistant question", ask)
    settle(page, 15)
    cam.keep("assistant answer", lambda: (hold(4), glide(page, 1200)))

    click_key(page, "nav_settings", 8)
    cam.keep("settings", lambda: (hold(4), glide(page, 2600), top(page)))

    click_key(page, "nav_dashboard", 10) if page.locator(".st-key-nav_dashboard button").count() else None
    settle(page, 8)
    cam.keep("closing dashboard", lambda: hold(6))

    page.close()
    cm.close()
    browser.close()

json.dump(CLIPS, open(f"{OUT}/clips.json", "w"), indent=1)
print("clips", len(CLIPS))
