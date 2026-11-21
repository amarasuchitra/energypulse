"""One continuous, uncut screen recording of the app on localhost (CI only), in dark mode.

A drawn mouse pointer follows every move and click so the video reads like a normal screen recording.
Email goes to a local test mail server (Mailpit), whose inbox is shown at the end.
"""
import os
import time
import traceback

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
INBOX = "http://localhost:8025"
EMAIL, PW = os.environ["SHOT_EMAIL"], os.environ["SHOT_PW"]
NEXT_READING = open("/tmp/next_reading.txt").read().strip()
SIZE = {"width": 1440, "height": 900}
os.makedirs("video2", exist_ok=True)
T0 = time.monotonic()

POINTER = """
(() => {
  if (window.top !== window) return;
  const add = () => {
    if (document.getElementById('__ptr')) return;
    const d = document.createElement('div'); d.id = '__ptr';
    d.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21 L8 16.5 L11.5 24 L14.5 22.6 L11 15.2 L17.5 15.2 Z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    Object.assign(d.style, {position:'fixed', left:'0px', top:'0px', zIndex:2147483647, pointerEvents:'none',
      transform:'translate(-3px,-2px)', filter:'drop-shadow(0 1px 2px rgba(0,0,0,.5))'});
    const r = document.createElement('div'); r.id = '__ring';
    Object.assign(r.style, {position:'fixed', width:'34px', height:'34px', borderRadius:'50%', zIndex:2147483646,
      pointerEvents:'none', border:'3px solid rgba(245,168,60,.95)', opacity:0, transform:'translate(-17px,-17px) scale(.4)',
      transition:'transform .35s ease-out, opacity .45s ease-out'});
    document.body.appendChild(r); document.body.appendChild(d);
  };
  window.__ptrMove = (x, y) => { add(); const d = document.getElementById('__ptr'); d.style.left = x + 'px'; d.style.top = y + 'px'; };
  window.__ptrClick = (x, y) => { add(); const r = document.getElementById('__ring');
    r.style.transition = 'none'; r.style.left = x + 'px'; r.style.top = y + 'px'; r.style.opacity = 1;
    r.style.transform = 'translate(-17px,-17px) scale(.4)'; r.offsetWidth;
    r.style.transition = 'transform .35s ease-out, opacity .45s ease-out';
    r.style.transform = 'translate(-17px,-17px) scale(1.3)'; r.style.opacity = 0; };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""

FIND_SCROLLER = """() => {
  let best = null, bestH = 0;
  for (const e of document.querySelectorAll('*')) {
    const st = getComputedStyle(e);
    if (!/(auto|scroll)/.test(st.overflowY)) continue;
    const extra = e.scrollHeight - e.clientHeight;
    if (extra > 50 && e.clientHeight > 300 && e.scrollHeight > bestH) { best = e; bestH = e.scrollHeight; }
  }
  window.__epScroller = best || document.scrollingElement;
  return window.__epScroller.scrollHeight - window.__epScroller.clientHeight - window.__epScroller.scrollTop;
}"""

pos = [720.0, 450.0]


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def step(label, fn):
    log("START " + label)
    try:
        fn()
    except Exception:
        log("FAILED " + label)
        traceback.print_exc()
    log("END " + label)


def hold(s):
    time.sleep(s)


def move(page, x, y, steps=22):
    x0, y0 = pos
    for i in range(1, steps + 1):
        k = i / steps
        k = k * k * (3 - 2 * k)
        cx, cy = x0 + (x - x0) * k, y0 + (y - y0) * k
        page.mouse.move(cx, cy)
        try:
            page.evaluate("([x, y]) => window.__ptrMove && window.__ptrMove(x, y)", [cx, cy])
        except Exception:
            pass
        time.sleep(0.016)
    pos[0], pos[1] = x, y


def point(page, loc):
    loc.scroll_into_view_if_needed(timeout=15_000)
    time.sleep(0.4)
    b = loc.bounding_box()
    move(page, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    time.sleep(0.25)


def click(page, loc, wait=1.0):
    point(page, loc)
    try:
        page.evaluate("([x, y]) => window.__ptrClick && window.__ptrClick(x, y)", pos)
    except Exception:
        pass
    loc.click()
    time.sleep(wait)


def type_into(page, loc, text, delay=75):
    click(page, loc, 0.3)
    loc.press_sequentially(text, delay=delay)
    time.sleep(0.5)


def settle(page, wait=2.0):
    time.sleep(wait)
    for _ in range(120):
        if not page.locator('[data-testid="stSpinner"], [data-testid="stSkeleton"]').count():
            break
        time.sleep(1)
    time.sleep(1.0)


def key_btn(page, key):
    return page.locator(f".st-key-{key} button").first


def nav(page, key, wait=6):
    click(page, key_btn(page, f"nav_{key}"), 0.5)
    settle(page, wait)


def tab(page, name, wait=3):
    click(page, page.get_by_role("tab", name=name).first, 0.5)
    settle(page, wait)


def scroll(page, px=None, speed=300, pause_every=600, pause=1.4):
    room = page.evaluate(FIND_SCROLLER)
    target = min(px, room) if px else room
    done = since = 0
    while done < target:
        page.mouse.wheel(0, speed / 30)
        done += speed / 30
        since += speed / 30
        time.sleep(1 / 30)
        if since >= pause_every:
            since = 0
            time.sleep(pause)
    time.sleep(1.0)


def to_top(page):
    page.evaluate(FIND_SCROLLER)
    page.evaluate("() => window.__epScroller.scrollTo({top: 0, behavior: 'smooth'})")
    time.sleep(1.2)


def home_frame(page):
    for f in page.frames:
        try:
            if f.locator("#member").count():
                return f
        except Exception:
            pass
    return None


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    ctx = browser.new_context(viewport=SIZE, record_video_dir="video2", record_video_size=SIZE,
                              timezone_id="Asia/Kolkata", color_scheme="dark")
    ctx.add_init_script(POINTER)
    page = ctx.new_page()
    page.goto(URL)
    page.wait_for_selector(".st-key-login_email input", timeout=240_000)
    settle(page, 2)

    # 1. Sign in -------------------------------------------------------------------------
    def sign_in():
        hold(3)
        type_into(page, page.locator(".st-key-login_email input"), EMAIL)
        type_into(page, page.locator(".st-key-login_pw input"), PW, delay=60)
        click(page, page.get_by_role("button", name="Sign In").first, 0.5)
        page.wait_for_selector(".st-key-dash_edit_appliances", timeout=300_000)
        settle(page, 10)
    step("sign in", sign_in)

    # 2. Real-time data on the dashboard --------------------------------------------------
    def realtime():
        move(page, 700, 300)
        hold(20)                                      # live tiles tick on their own
        scroll(page, 700)
        hold(10)
    step("real-time dashboard", realtime)

    # 3. Add a meter reading and compare it with what the app measured ------------------
    def meter():
        to_top(page)
        exp = page.get_by_text("Add a meter reading").first
        box = page.locator('input[aria-label="Number on the meter (kWh)"]').first
        if not box.is_visible():
            click(page, exp, 1.0)
        type_into(page, box, NEXT_READING, delay=140)
        hold(1)
        click(page, key_btn(page, "mr_save"), 0.5)
        settle(page, 4)
        point(page, page.get_by_text("Units so far").first)
        hold(8)
        scroll(page, 500)
        hold(6)
    step("meter reading", meter)

    # 4. Add family members ------------------------------------------------------------
    def family():
        nav(page, "household", 6)
        hold(2)
        for name, rel, mail in (("Ravi Rao", "Spouse", "ravi.rao@example.com"),
                                ("Meera Rao", "Daughter", "meera.rao@example.com")):
            type_into(page, page.locator(".st-key-fam_new_name input"), name)
            sel = page.locator(".st-key-fam_new_rel")
            click(page, sel, 0.8)
            opt = page.get_by_role("option", name=rel).first
            if opt.count():
                click(page, opt, 0.6)
            else:
                page.keyboard.press("Escape")
            type_into(page, page.locator(".st-key-fam_new_email input"), mail)
            click(page, page.locator('[data-testid="stFormSubmitButton"] button').first, 0.5)
            settle(page, 3)
            hold(3)
        to_top(page)
        hold(4)
        scroll(page, 600)
        hold(3)
    step("family members", family)

    # 5. Switch appliances on and off in the 3-D home, as a family member ---------------
    def switches():
        nav(page, "dashboard", 10)
        f = home_frame(page)
        if f is None:
            settle(page, 10)
            f = home_frame(page)
        iframe = f.frame_element()
        iframe.scroll_into_view_if_needed()
        hold(2)
        page.evaluate(FIND_SCROLLER)
        b = iframe.bounding_box()
        page.evaluate("(y) => window.__epScroller.scrollBy({top: y - 20, behavior: 'smooth'})", b["y"])
        hold(3)
        ib = iframe.bounding_box()

        def in_frame(sel):
            el = f.locator(sel).first
            bb = el.bounding_box()
            return el, bb

        # choose who is switching
        el, bb = in_frame("#member")
        move(page, bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.evaluate("([x, y]) => window.__ptrClick(x, y)", pos)
        opts = f.locator("#member option").all_inner_texts()
        pick = next((o for o in opts if "Ravi" in o), opts[-1] if opts else None)
        if pick:
            el.select_option(label=pick)
        hold(2)

        def flip(key, confirm=False):
            sw = f.locator(f'#list li[data-key="{key}"] .switch').first
            if not sw.count():
                return
            sw.scroll_into_view_if_needed()
            hold(0.5)
            bb = sw.bounding_box()
            move(page, bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
            page.evaluate("([x, y]) => window.__ptrClick(x, y)", pos)
            sw.click()
            hold(1.5)
            if confirm and f.locator("#confirm").is_visible():
                yes = f.locator("#confirmYes")
                bb = yes.bounding_box()
                move(page, bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
                page.evaluate("([x, y]) => window.__ptrClick(x, y)", pos)
                yes.click()
            hold(5)

        keys = f.locator("#list li").evaluate_all("els => els.map(e => [e.dataset.key, e.classList.contains('is-on')])")
        log(f"switches {keys}")
        on_now = [k for k, on in keys if on and k not in ("fridge",)]
        off_now = [k for k, on in keys if not on and k not in ("fridge",)]
        for k in off_now[:3]:
            flip(k)                                   # switch on
        hold(6)
        for k in (on_now[:1] + off_now[:1]):
            flip(k, confirm=True)                     # switch off, with confirmation
        hold(10)
        move(page, ib["x"] + ib["width"] * 0.35, ib["y"] + ib["height"] * 0.35)
        hold(6)
    step("switch appliances", switches)

    # 6. Add appliances: by photo and from the list -----------------------------------
    def appliances():
        nav(page, "appliances", 8)
        tab(page, "My Devices", 3)
        hold(2)
        zone = page.locator('[class*="st-key-am_drop_zone"]').first
        point(page, zone)
        page.locator('[class*="st-key-am_drop_zone"] input[type=file]').first.set_input_files(
            os.path.join("tests", "fixtures", "ceiling_fan.jpg"))
        settle(page, 25)
        hold(4)
        save = page.locator('[class*="st-key-am_drop_save_"] button').first
        if save.count():
            click(page, save, 0.5)
            settle(page, 5)
        hold(3)
        cat = page.locator(".st-key-am_cat")
        click(page, cat, 0.8)
        opt = page.get_by_role("option", name="Laundry").first
        if not opt.count():
            opt = page.get_by_role("option").nth(2)
        click(page, opt, 1.5)
        click(page, page.locator(".st-key-am_type"), 0.8)
        opt = page.get_by_role("option").first
        click(page, opt, 1.0)
        click(page, key_btn(page, "am_add"), 0.5)
        settle(page, 5)
        hold(3)
        scroll(page, 1400)
        hold(4)
        to_top(page)
        tab(page, "What Meter Shows", 6)
        hold(3)
        scroll(page, 1500)
        to_top(page)
        tab(page, "Where Power Goes", 6)
        hold(3)
        scroll(page, 1500)
        hold(2)
    step("add appliances", appliances)

    # 7. Optimisation solution --------------------------------------------------------
    def optimize():
        nav(page, "optimize", 10)
        hold(4)
        scroll(page, 3200, speed=260)
        hold(3)
        to_top(page)
        tab(page, "Monthly Goals", 5)
        hold(3)
        scroll(page, 1000)
        to_top(page)
        tab(page, "Upgrade Calculator", 5)
        hold(3)
        scroll(page, 900)
        hold(2)
    step("optimize", optimize)

    # 8. Email: send a test e-mail and run the alert checks -------------------------
    def email():
        nav(page, "bills_safety", 8)
        tab(page, "Alerts", 6)
        hold(3)
        scroll(page, 700)
        hold(2)
        rec = page.locator(".st-key-notif_test_recipient")
        click(page, rec, 0.8)
        opt = page.get_by_role("option", name="ravi.rao@example.com").first
        if opt.count():
            click(page, opt, 1.0)
        else:
            page.keyboard.press("Escape")
        click(page, key_btn(page, "notif_send_test"), 0.5)
        settle(page, 4)
        hold(5)
        run = key_btn(page, "notif_run_checks")
        if run.count():
            click(page, run, 0.5)
            settle(page, 6)
            hold(4)
        scroll(page, 1500)
        hold(4)
    step("email", email)

    # 9. Assistant ---------------------------------------------------------------------
    def assistant():
        nav(page, "household", 6)
        tab(page, "Energy Assistant", 4)
        type_into(page, page.locator(".st-key-chat_input input"), "How much will next month cost?")
        click(page, page.get_by_role("button", name="Ask").first, 0.5)
        settle(page, 12)
        hold(6)
    step("assistant", assistant)

    # 10. Back to the live dashboard -------------------------------------------------
    def closing():
        nav(page, "dashboard", 8)
        hold(15)
    step("dashboard again", closing)

    # 11. The inbox of the test mail server ------------------------------------------
    def inbox():
        page.goto(INBOX)
        hold(4)
        first = page.locator(".message, a.message, [class*=message]").first
        if not first.count():
            first = page.get_by_text("EnergyPulse").first
        click(page, first, 1.0)
        hold(10)
        scroll(page, 800)
        hold(4)
        page.go_back()
        hold(2)
        items = page.locator(".message, a.message, [class*=message]")
        if items.count() > 1:
            click(page, items.nth(1), 1.0)
            hold(10)
    step("inbox", inbox)

    page.close()
    ctx.close()
    browser.close()
log("done")
