"""Sign in to the app on localhost and save screenshots of each feature (CI only)."""
import os
import time
import traceback

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = "shots"
EMAIL, PW = os.environ["SHOT_EMAIL"], os.environ["SHOT_PW"]
os.makedirs(OUT, exist_ok=True)

SCROLL_JS = """(y) => {
  const c = [...document.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, .main')]
    .find(e => e.scrollHeight > e.clientHeight + 10) || document.scrollingElement;
  if (y === 'h') return c.scrollHeight;
  c.scrollTo(0, y); return c.scrollTop;
}"""


def settle(page, wait=4.0):
    time.sleep(wait)
    for _ in range(90):
        busy = page.locator('[data-testid="stSpinner"], [data-testid="stSkeleton"]').count()
        if not busy:
            break
        time.sleep(1)
    time.sleep(1.5)


def shot(page, name):
    page.screenshot(path=f"{OUT}/{name}.png")
    print("saved", name)


def scroll_series(page, prefix, n=8, step=780):
    page.mouse.move(1000, 520)
    for _ in range(12):                                # back to the top
        page.mouse.wheel(0, -3000)
    time.sleep(1.0)
    last = None
    for i in range(n):
        data = page.screenshot()
        if data == last:
            break
        with open(f"{OUT}/{prefix}_{i + 1:02d}.png", "wb") as fh:
            fh.write(data)
        print("saved", prefix, i + 1)
        last = data
        page.mouse.wheel(0, step)
        time.sleep(1.6)
    for _ in range(12):
        page.mouse.wheel(0, -3000)
    time.sleep(0.8)


def click_key(page, key, wait=4.0):
    page.locator(f".st-key-{key} button").first.click()
    settle(page, wait)


def tab(page, name, wait=4.0):
    page.get_by_role("tab", name=name).first.click()
    settle(page, wait)


def step(label, fn):
    try:
        fn()
    except Exception:
        print("FAILED", label)
        traceback.print_exc()


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5,
                              timezone_id="Asia/Kolkata", color_scheme="dark")
    page = ctx.new_page()
    page.goto(URL)
    page.wait_for_selector(".st-key-login_email input", timeout=240_000)
    settle(page, 3)
    shot(page, "01_signin")

    # A guest goes through setup (no account needed).
    g = ctx.new_page()

    def guest():
        g.goto(URL)
        g.wait_for_selector(".st-key-guest_btn button", timeout=120_000)
        click_key(g, "guest_btn", 6)
        try:
            g.get_by_role("button", name="Light mode").first.click(); settle(g, 4)
        except Exception:
            pass
        shot(g, "02_setup_home")
        click_key(g, "ob_next1", 5)
        shot(g, "03_setup_appliances")
        g.get_by_role("tab", name="Kitchen").first.click()
        time.sleep(2)
        shot(g, "03b_setup_appliances_kitchen")
    step("guest setup", guest)
    g.close()

    page.locator(".st-key-login_email input").fill(EMAIL)
    page.locator(".st-key-login_pw input").fill(PW)
    page.get_by_role("button", name="Sign In").first.click()
    page.wait_for_selector(".st-key-dash_edit_appliances", timeout=300_000)
    settle(page, 25)                                  # models, 3-D home and the live tiles

    def light_mode():
        page.get_by_role("button", name="Light mode").first.click()
        settle(page, 12)
    step("light mode", light_mode)
    settle(page, 10)
    scroll_series(page, "04_dashboard", 14)

    def edit():
        click_key(page, "dash_edit_appliances", 6)
        shot(page, "05_edit_appliances")
        page.evaluate(SCROLL_JS, 700)
        time.sleep(1.5)
        shot(page, "05b_edit_appliances_list")
        click_key(page, "ap_cancel", 8)
    step("edit appliances", edit)

    def appliances():
        click_key(page, "nav_appliances", 10)
        scroll_series(page, "06_appliances_devices", 4)
        tab(page, "What Meter Shows", 8)
        scroll_series(page, "07_appliances_meter", 4)
        tab(page, "Where Power Goes", 8)
        scroll_series(page, "08_appliances_power", 4)
    step("appliances", appliances)

    def photo():
        click_key(page, "nav_appliances", 6)
        tab(page, "My Devices", 4)
        up = page.locator('[class*="st-key-am_drop_zone"] input[type=file]').first
        up.set_input_files(os.path.join("tests", "fixtures", "ceiling_fan.jpg"))
        settle(page, 40)
        scroll_series(page, "09_photo_recognition", 4)
    step("photo", photo)

    def optimize():
        click_key(page, "nav_optimize", 10)
        scroll_series(page, "10_optimize_tips", 5)
        tab(page, "Monthly Goals", 6)
        shot(page, "11_optimize_goals")
        tab(page, "Upgrade Calculator", 6)
        shot(page, "12_optimize_upgrades")
    step("optimize", optimize)

    def bills():
        click_key(page, "nav_bills_safety", 10)
        scroll_series(page, "13_bills", 4)
        tab(page, "Safety", 6)
        scroll_series(page, "14_safety", 3)
        tab(page, "Alerts", 6)
        scroll_series(page, "15_alerts", 6)
    step("bills", bills)

    def household():
        click_key(page, "nav_household", 8)
        shot(page, "16_family")
        tab(page, "Energy Assistant", 5)
        page.locator(".st-key-chat_input input").fill("How much will next month cost?")
        page.get_by_role("button", name="Ask").first.click()
        settle(page, 20)
        scroll_series(page, "17_assistant", 3)
    step("household", household)

    def settings():
        click_key(page, "nav_settings", 8)
        scroll_series(page, "18_settings", 7)
    step("settings", settings)

    def light():
        lp = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5,
                                 timezone_id="Asia/Kolkata", color_scheme="light").new_page()
        lp.goto(URL)
        lp.wait_for_selector(".st-key-login_email input", timeout=120_000)
        settle(lp, 3)
        shot(lp, "19_signin_light")
    step("light", light)
    browser.close()
