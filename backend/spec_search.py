"""
Looking up an appliance's specification on the web
==================================================
When the user knows (or the rating-plate photo shows) the brand and model,
the app searches the web for it, reads the top result pages, picks out the
power, voltage, current, star rating and capacity, and keeps a figure only
where sources agree.  No paid service or key is used: it reads DuckDuckGo's
plain HTML results page, with Bing's as a fallback.

Search results can be wrong or about a different model, so the figures are
shown with their sources and the user confirms them before they are used.

    found = look_up("LG GL-B201 refrigerator")
    found["figures"]   -> {"rated_w": 120.0, "voltage_v": 230.0, ...}
    found["agree"]     -> {"rated_w": 3, ...}   how many sources gave that value
    found["sources"]   -> [{"title", "url", "found": {...}}, ...]
"""

import re
from collections import Counter
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

from nameplate import parse_plate

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36", "Accept-Language": "en-IN,en;q=0.8"}
TIMEOUT = 7
MAX_PAGES = 5
_SKIP_SITES = ("youtube.com", "facebook.com", "instagram.com", "pinterest.", "x.com", "twitter.com")
_KEYWORDS = re.compile(r"power|watt|consumption|input|voltage|volt|current|amp|star|capacity|litre|liter|"
                       r"ton\b|hp\b|rated|energy|kw", re.I)


def _get(url: str, **kw) -> Optional[str]:
    import requests
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kw)
        if r.status_code == 200 and "html" in r.headers.get("content-type", "html"):
            return r.text
    except Exception:
        return None
    return None


def _post(url: str, data: dict) -> Optional[str]:
    import requests
    try:
        r = requests.post(url, data=data, headers=HEADERS, timeout=TIMEOUT)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def search(query: str, fetch_get: Callable = _get, fetch_post: Callable = _post) -> List[dict]:
    """Result titles and links for `query`, from DuckDuckGo, else Bing."""
    from bs4 import BeautifulSoup
    results = []
    html = fetch_post("https://html.duckduckgo.com/html/", {"q": query, "kl": "in-en"})
    if html:
        for a in BeautifulSoup(html, "html.parser").select("a.result__a"):
            href = a.get("href", "")
            if "uddg=" in href:                              # DuckDuckGo wraps the real link
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [""])[0])
            results.append({"title": a.get_text(" ", strip=True), "url": href})
    if not results:
        html = fetch_get("https://www.bing.com/search", params={"q": query, "setlang": "en-IN"})
        if html:
            for a in BeautifulSoup(html, "html.parser").select("li.b_algo h2 a"):
                results.append({"title": a.get_text(" ", strip=True), "url": a.get("href", "")})
    seen, clean = set(), []
    for r in results:
        url = r["url"]
        if not url.startswith("http") or any(s in url for s in _SKIP_SITES) or url in seen:
            continue
        seen.add(url)
        clean.append(r)
    return clean


def page_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()
    # Table cells become "label value" lines, which is how spec tables read.
    for row in soup.find_all("tr"):
        row.replace_with(" ".join(c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])) + "\n")
    return soup.get_text("\n", strip=True)


def figures_in(text: str, model: str = "") -> dict:
    """Specification figures in one page: only lines that talk about power, voltage and the like."""
    lines = [l for l in text.splitlines() if len(l) < 220 and _KEYWORDS.search(l)]
    found = parse_plate("\n".join(lines[:400]))
    found.pop("lines", None)
    found.pop("voltage_text", None)
    return {k: v for k, v in found.items() if k in ("rated_w", "voltage_v", "current_a", "frequency_hz", "star",
                                                    "capacity", "from_current", "from_hp")}


def look_up(query: str, fetch_get: Callable = _get, fetch_post: Callable = _post,
            max_pages: int = MAX_PAGES) -> dict:
    query = " ".join(query.split())[:120]
    results = search(f"{query} specifications power consumption watts", fetch_get, fetch_post)
    sources = []
    for r in results[:max_pages]:
        html = fetch_get(r["url"])
        if not html:
            continue
        found = figures_in(page_text(html), query)
        if any(k in found for k in ("rated_w", "current_a", "star", "capacity")):
            sources.append({**r, "found": found})
    votes: Dict[str, Counter] = {}
    for s in sources:
        for k, v in s["found"].items():
            if k in ("from_current", "from_hp"):
                continue
            votes.setdefault(k, Counter())[round(v) if k == "rated_w" else v] += 1
    figures, agree = {}, {}
    for k, counter in votes.items():
        value, count = counter.most_common(1)[0]
        figures[k], agree[k] = value, count
    return {"query": query, "figures": figures, "agree": agree, "sources": sources,
            "searched": len(results), "ok": bool(results)}
