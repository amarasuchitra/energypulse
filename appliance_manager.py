"""
Describing an appliance, and the "My appliances" page
=====================================================
Three ways to describe each appliance, because many people do not know the
model of an old fridge or pump:

  I know the model class      pick it from the catalogue
  Not sure                    pick the size and roughly how old it is; the
                              specification is estimated
  Rating plate / own figures  photograph the rating plate (read for free on
                              the server by Tesseract) or type the figures

Photos are stored encrypted for the household.  Appliances can be added,
changed or removed at any time from the My appliances page; changes are
saved straight away and the 3D home and every figure follow them.
"""

import copy
import json

import streamlit as st

import catalog
from nameplate import PHOTO_TYPES, load_photo, parse_plate, read_plate, store_photo


def spec_table(spec: dict) -> str:
    rows = "".join(f"<tr><td class='n'>{k}</td><td class='v'>{v}</td></tr>" for k, v in catalog.spec_lines(spec))
    note = f"<p style='color:var(--mist);font-size:.78rem;margin:.4rem 0 0'>{spec['note']}</p>" if spec.get("note") else ""
    source = {"catalogue": "Typical figures for this class of model. The rating plate on your own appliance gives its exact values.",
              "estimated": "Estimated, because the model is not known. A photo of the rating plate makes it exact.",
              "rating plate": "From the rating plate, as you checked it.",
              "web": "Looked up on the web and checked by you.",
              "entered": "As you entered them."}.get(spec.get("source", "catalogue"), "")
    return (f"<table class='ep-rows'>{rows}</table>{note}"
            f"<p style='color:var(--faint);font-size:.74rem;margin:.35rem 0 0'>{source}</p>")


def _say(field: str, value) -> str:
    """One figure in plain words (formatted only for the field it belongs to)."""
    try:
        if field == "rated_w":
            return f"power {float(value):.0f} W"
        if field == "voltage_v":
            return f"voltage {float(value):.0f} V"
        if field == "current_a":
            return f"current {float(value):g} A"
        if field == "frequency_hz":
            return f"{value} Hz"
        if field == "star":
            return f"{value}-star"
    except (TypeError, ValueError):
        pass
    return {"capacity": f"capacity {value}", "model_no": f"model {value}"}.get(field, f"{field} {value}")


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def _look_up(query: str) -> dict:
    from spec_search import look_up
    return look_up(query)


_FIELD_KEYS = (("rated_w", "_w"), ("voltage_v", "_v"), ("current_a", "_a"), ("star", "_star"),
               ("capacity", "_cap"), ("model_no", "_model"))


def _fill(item: dict, prefix: str, figures: dict) -> None:
    """Put figures into the item and into the boxes drawn further down the page."""
    item["spec_mode"] = "own"
    item["own"] = {**(item.get("own") or {}), **figures}
    for field, suffix in _FIELD_KEYS:
        if field in figures:
            value = figures[field]
            st.session_state[f"{prefix}{suffix}"] = float(value) if suffix in ("_w", "_v", "_a") else value


def _web_search(item: dict, prefix: str) -> bool:
    """Search the web for the brand and model, show what sources agree on, and let the user take it."""
    own = item.get("own") or {}
    suggestion = " ".join(x for x in [own.get("model_no", ""), item.get("name", "")] if x).strip()
    if f"{prefix}_q" not in st.session_state:
        st.session_state[f"{prefix}_q"] = suggestion
    c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
    query = c1.text_input("Brand and model", key=f"{prefix}_q",
                          placeholder="e.g. LG GL-B201 refrigerator, or the model number on the plate",
                          help="Written on the rating plate, the bill of purchase or the box.")
    if c2.button("Search the web", key=f"{prefix}_search", disabled=len(query.strip()) < 3, width="stretch"):
        with st.spinner("Searching the web and reading the specification pages..."):
            st.session_state[f"{prefix}_web"] = _look_up(query.strip())
        return True
    found = st.session_state.get(f"{prefix}_web")
    if not found:
        return False
    if not found["ok"]:
        st.warning("The web search could not be reached just now. Try again in a minute, or use a photo of the plate.")
        return False
    if not found["figures"]:
        st.info(f"Read {len(found['sources'])} of {found['searched']} results for \"{found['query']}\" but found no "
                "specification figures. Try the exact model number from the rating plate, or use a photo of it.")
        return False
    n = len(found["sources"])
    rows = "".join(f"<tr><td class='n'>{_say(k, v)}</td><td class='v'>{found['agree'][k]} of {n} "
                   f"source{'s' if n != 1 else ''}</td></tr>" for k, v in found["figures"].items())
    st.markdown(f"<b>Found online for \"{found['query']}\"</b><table class='ep-rows'>{rows}</table>", unsafe_allow_html=True)
    st.caption("Sources: " + " · ".join(f"[{(s['title'] or s['url'])[:60]}]({s['url']})" for s in found["sources"]))
    weak = [k for k, c in found["agree"].items() if c < 2 and n > 1]
    if weak:
        st.caption("Only one source gave: " + ", ".join(_say(k, found["figures"][k]) for k in weak)
                   + ". Check it against your appliance.")
    if st.button("Use these figures", key=f"{prefix}_use_web", type="primary"):
        figures = {k: v for k, v in found["figures"].items() if k in dict(_FIELD_KEYS)}
        _fill(item, prefix, figures)
        item["own"]["source_urls"] = [s["url"] for s in found["sources"]][:5]
        item["own"]["looked_up"] = found["query"]
        st.session_state.pop(f"{prefix}_web", None)
        return True
    return False


def _photos(item: dict, household_id: str, prefix: str) -> bool:
    """Photo upload, thumbnails and reading the rating plate.  Returns True when the page should refresh."""
    item.setdefault("photos", [])
    upload = st.file_uploader("Photo of the appliance or its rating plate", type=PHOTO_TYPES,
                              key=f"{prefix}_photo",
                              help="On a phone this opens the camera. The rating plate is the sticker or metal "
                                   "plate with the power, voltage and current, usually at the back or the side.")
    if upload is not None:
        data = upload.getvalue()
        pid = store_photo(household_id, data)
        if pid not in item["photos"]:
            item["photos"].append(pid)
            st.session_state[f"{prefix}_newest"] = pid
            return True
    if item["photos"]:
        cols = st.columns(min(4, len(item["photos"])) + 1)
        for col, pid in zip(cols, list(item["photos"])[-4:]):
            data = load_photo(household_id, pid)
            if data:
                col.image(data, width=110)
            if col.button("Remove", key=f"{prefix}_rm_{pid}"):
                item["photos"].remove(pid)
                return True
        newest = st.session_state.get(f"{prefix}_newest") or item["photos"][-1]
        if st.button("Read the rating plate from the photo", key=f"{prefix}_ocr", type="primary"):
            data = load_photo(household_id, newest)
            text, ok = read_plate(data or b"")
            if ok is False:
                st.session_state[f"{prefix}_ocr_msg"] = ("error", "Text reading is not available on this server. Type the figures instead.")
            else:
                found = parse_plate(text)
                figures = {k: v for k, v in found.items() if k in ("rated_w", "voltage_v", "current_a", "frequency_hz",
                                                                     "star", "capacity", "model_no")}
                st.session_state[f"{prefix}_ocr_text"] = text.strip()
                if figures:
                    _fill(item, prefix, figures)            # the boxes below are drawn after this point
                    if figures.get("model_no"):
                        st.session_state[f"{prefix}_q"] = f"{figures['model_no']} {item.get('name', '')}".strip()
                    how = (" Power was worked out from the current printed on the plate." if found.get("from_current")
                           else " Power was worked out from the motor size (HP)." if found.get("from_hp") else "")
                    st.session_state[f"{prefix}_ocr_msg"] = (
                        "success", "Read from the plate: " + ", ".join(_say(k, v) for k, v in figures.items())
                        + "." + how + " Check them below and correct anything misread.")
                else:
                    st.session_state[f"{prefix}_ocr_msg"] = (
                        "warning", "No figures could be read. Try a sharper, closer photo of the plate in good light, "
                                   "or type the figures below.")
            return True
        msg = st.session_state.get(f"{prefix}_ocr_msg")
        if msg:
            getattr(st, msg[0])(msg[1])
        if st.session_state.get(f"{prefix}_ocr_text"):
            with st.expander("What the camera read"):
                st.text(st.session_state[f"{prefix}_ocr_text"][:1500])
    return False


def _own_figures(item: dict, prefix: str) -> None:
    own = item.setdefault("own", {})
    t = catalog.get_type(item.get("type_key", ""))
    base = catalog.specs(item.get("model_id")) if item.get("model_id") else (
        catalog.specs(catalog.default_model(t.key).id) if t else None)
    c1, c2, c3 = st.columns(3)
    own["rated_w"] = c1.number_input("Rated power (W)", 0.0, 20000.0, float(own.get("rated_w") or (base or {}).get("rated_w", 0)),
                                     10.0, key=f"{prefix}_w", help="Printed as W, watts, input power or power consumption.")
    own["voltage_v"] = c2.number_input("Voltage (V)", 100.0, 260.0, float(own.get("voltage_v") or 230.0), 5.0, key=f"{prefix}_v")
    own["current_a"] = c3.number_input("Current (A)", 0.0, 60.0, float(own.get("current_a") or 0.0), 0.1, key=f"{prefix}_a",
                                       help="Leave at 0 to work it out from the power.")
    c4, c5, c6 = st.columns(3)
    stars = [None, 1, 2, 3, 4, 5]
    own["star"] = c4.selectbox("Star rating", stars, index=stars.index(own.get("star")) if own.get("star") in stars else 0,
                               format_func=lambda s: "Not shown" if s is None else f"{s}-star", key=f"{prefix}_star")
    own["capacity"] = c5.text_input("Capacity", own.get("capacity", ""), key=f"{prefix}_cap",
                                    placeholder="e.g. 190 L, 1.5 ton, 1 HP")
    own["model_no"] = c6.text_input("Brand and model (optional)", own.get("model_no", ""), key=f"{prefix}_model")
    if not own.get("current_a"):
        own.pop("current_a", None)


def spec_editor(item: dict, household_id: str, prefix: str) -> bool:
    """Choose how this appliance is described.  Mutates `item`; returns True when the page should refresh."""
    return _spec_editor(item, household_id, prefix)


def _spec_editor(item: dict, household_id: str, prefix: str) -> bool:
    t = catalog.get_type(item.get("type_key", ""))
    modes = list(catalog.SPEC_MODES) if t else ["own"]
    current = item.get("spec_mode", "catalog" if t else "own")
    mode = st.radio("How do you want to describe it?", modes, index=modes.index(current) if current in modes else 0,
                    format_func=catalog.SPEC_MODES.get, key=f"{prefix}_mode", horizontal=True)
    item["spec_mode"] = mode
    if mode == "catalog":
        labels, ids = [m.label for m in t.models], [m.id for m in t.models]
        pick = st.selectbox("Model class", labels, index=ids.index(item["model_id"]) if item.get("model_id") in ids else 0,
                            key=f"{prefix}_model_class")
        item["model_id"] = ids[labels.index(pick)]
    elif mode == "estimate":
        options = catalog.sizes(t.key)
        c1, c2 = st.columns(2)
        item["size"] = c1.selectbox("Size", options, index=options.index(item["size"]) if item.get("size") in options else
                                    min(1, len(options) - 1), key=f"{prefix}_size")
        item["age_years"] = c2.slider("About how old is it? (years)", 0, 25, int(item.get("age_years", 8)), key=f"{prefix}_age")
    refresh = False
    if mode == "own":
        st.markdown("<div style='font-size:.84rem;color:var(--mist);margin:.2rem 0 .3rem'><b style='color:var(--plaster)'>"
                    "Look it up</b> by brand and model, <b style='color:var(--plaster)'>read the rating plate</b> from a "
                    "photo, or type the figures. Every figure stays editable.</div>", unsafe_allow_html=True)
        refresh = _photos(item, household_id, prefix)
        refresh = _web_search(item, prefix) or refresh
        _own_figures(item, prefix)
    else:
        with st.expander("Add a photo (optional)"):
            refresh = _photos(item, household_id, prefix) or refresh
    spec = catalog.effective_specs(item)
    if spec:
        with st.expander(f"Specification: {spec['rated_w']:.0f} W, {spec['voltage_v']:.0f} V, {spec['current_a']:.2f} A"
                         + (" (estimated)" if spec.get("source") == "estimated" else "")):
            st.markdown(spec_table(spec), unsafe_allow_html=True)
    return refresh


# ------------------------------------------------------------ drop a photo
# Words on a rating plate or a product label that say what the appliance is.
_PLATE_WORDS = [
    ("fridge", r"refrigerat|fridge|freezer|frost"), ("ac", r"air ?condition|split|cooling capacity|\binverter ac\b|ton\b|tr\b"),
    ("washing_machine", r"washing|washer|wash load|spin"), ("geyser", r"water heater|geyser|storage heater|instant heater"),
    ("water_pump", r"pump|monoblock|submersible|head\s*\(?m|discharge|\bhp\b"), ("microwave", r"microwave|oven|grill"),
    ("tv", r"television|led tv|\btv\b|uhd|4k|smart tv"), ("fan", r"ceiling fan|\bfan\b|sweep|bldc"),
    ("cooler", r"cooler"), ("iron", r"\biron\b"), ("induction", r"induction|cooktop"), ("mixer", r"mixer|grinder|juicer"),
    ("kettle", r"kettle"), ("router", r"router|wi-?fi"), ("purifier", r"purifier|\bro\b"), ("chimney", r"chimney|hood"),
    ("heater", r"room heater|fan heater|oil heater"), ("dishwasher", r"dishwasher"), ("vacuum", r"vacuum"),
]


def guess_type(text: str, figures: dict) -> str:
    """Which appliance a plate most likely belongs to, from its words, else from its size."""
    import re as _re
    low = (text or "").lower()
    for key, pattern in _PLATE_WORDS:
        if _re.search(pattern, low):
            return key
    cap, watts = str(figures.get("capacity", "")), float(figures.get("rated_w") or 0)
    if cap.endswith(" L") and watts and watts < 400:
        return "fridge"
    if cap.endswith(" L") and watts >= 1000:
        return "geyser"
    if cap.endswith(" kg"):
        return "washing_machine"
    if cap.endswith(" HP"):
        return "water_pump"
    if cap.endswith(" ton"):
        return "ac"
    return ""


def render_drop_zone(db, household_id: str, notify, key: str = "drop", compact: bool = False) -> None:
    """
    Drop (or choose) a photo of any appliance or its rating plate.  The plate is
    read, the appliance is recognised from its words, and the user decides
    whether it is a new appliance or belongs to one already listed.
    """
    hd = st.session_state.home_details
    items = hd.setdefault("appliances", [])
    round_ = st.session_state.setdefault(f"{key}_round", 0)       # a fresh box after each save
    with st.container(key=f"{key}_zone"):
        upload = st.file_uploader("Drop a photo of an appliance or its rating plate here",
                                  type=PHOTO_TYPES, key=f"{key}_file_{round_}",
                                  help="Drag a photo from your computer onto this box, or click to choose one. "
                                       "On a phone it opens the camera.")
    if upload is None:
        st.session_state.pop(f"{key}_result", None)
        return
    data = upload.getvalue()
    pid = store_photo(household_id, data)
    result = st.session_state.get(f"{key}_result")
    if not result or result["pid"] != pid:
        with st.spinner("Reading the photo..."):
            text, ok = read_plate(data)
        figures = parse_plate(text) if ok else {}
        result = {"pid": pid, "ok": ok, "text": text.strip(),
                  "figures": {k: v for k, v in figures.items() if k in ("rated_w", "voltage_v", "current_a", "frequency_hz",
                                                                       "star", "capacity", "model_no")},
                  "guess": guess_type(text, figures)}
        st.session_state[f"{key}_result"] = result

    with st.container(border=True):
        left, right = st.columns([1, 2.4], gap="medium")
        left.image(data, width=220)
        with right:
            fig = result["figures"]
            if fig:
                st.markdown("**Read from the photo:** " + ", ".join(_say(k, v) for k, v in fig.items()))
            elif result["ok"] is False:
                st.caption("Text reading is not available here. Choose the appliance below; you can type its figures later.")
            else:
                st.caption("No figures could be read from this photo. It is kept with the appliance; for exact figures "
                           "try a close, sharp photo of the rating plate.")
            types = [t for t in catalog.CATALOG]
            names = [t.name for t in types]
            guess = next((i for i, t in enumerate(types) if t.key == result["guess"]), 0)
            kind = st.selectbox("This is a" + (" (recognised from the plate)" if result["guess"] else ""), names,
                                index=guess, key=f"{key}_type_{pid}")
            t = types[names.index(kind)]
            same = [i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type_key") == t.key]
            choices = ["Add it as a new appliance"] + [f"It is my {items[i]['name']} ({n})" for n, i in enumerate(same, 1)]
            where = st.radio("Where does it go?", choices, index=1 if same else 0, key=f"{key}_where_{pid}")
            if st.button("Save", type="primary", key=f"{key}_save_{pid}"):
                if where == choices[0]:
                    item = catalog.item_for(t.key)
                    items.append(item)
                    st.session_state["am_open"] = len(items) - 1
                else:
                    item = items[same[choices.index(where) - 1]]
                    st.session_state["am_open"] = items.index(item)
                item.setdefault("photos", [])
                if pid not in item["photos"]:
                    item["photos"].append(pid)
                if fig.get("rated_w"):
                    item["spec_mode"] = "own"
                    item["own"] = {**(item.get("own") or {}), **fig}
                _save(db, household_id)
                st.session_state.pop(f"{key}_result", None)
                st.session_state[f"{key}_round"] = round_ + 1
                notify("success", f"Photo saved with your {item['name'].lower()}"
                       + (" and its figures were filled in from the plate." if fig.get("rated_w") else "."))
                st.rerun()
            if result["text"]:
                with st.expander("What the camera read"):
                    st.text(result["text"][:1500])


# ---------------------------------------------------------------- the page
def _save(db, household_id: str) -> None:
    hd = st.session_state.home_details
    try:
        db.save_home_details(household_id, hd)
    except Exception:
        pass
    st.session_state.pop("home_state", None)          # the 3D home is rebuilt from the new list


def render_page(db, household_id: str, section, notify) -> None:
    hd = st.session_state.home_details
    items = hd.setdefault("appliances", [])
    before = json.dumps(items, sort_keys=True, default=str)

    section("Add an appliance from a photo")
    render_drop_zone(db, household_id, notify, key="am_drop")
    section("Or pick from the list")
    c1, c2, c3 = st.columns([1.2, 1.6, 0.8], vertical_alignment="bottom")
    category = c1.selectbox("Category", catalog.CATEGORIES + ["Something else"], key="am_cat")
    if category == "Something else":
        name = c2.text_input("Name", placeholder="For example: sewing machine", key="am_custom")
        if c3.button("Add", type="primary", key="am_add_custom", disabled=not name.strip(), width="stretch"):
            items.append({"name": name.strip()[:40], "type": "Other", "usage": "Medium", "icon": "", "spec_mode": "own", "own": {}})
            st.session_state["am_open"] = len(items) - 1
            _save(db, household_id)
            notify("success", f"{name.strip()} added. Add a photo of its rating plate or type its figures below.")
            st.rerun()
    else:
        types = catalog.types_in(category)
        pick = c2.selectbox("Appliance", [t.name for t in types], key="am_type")
        if c3.button("Add", type="primary", key="am_add", width="stretch"):
            t = next(t for t in types if t.name == pick)
            items.append(catalog.item_for(t.key))
            st.session_state["am_open"] = len(items) - 1
            _save(db, household_id)
            notify("success", f"{t.name} added to your home.")
            st.rerun()

    section(f"Your appliances ({len(items)})")
    st.caption("Changes are saved straight away. The 3D home, the forecast and the advice all follow this list.")
    for i, item in enumerate(list(items)):
        if not isinstance(item, dict):
            item = items[i] = {"name": str(item), "type": "Other", "usage": "Medium"}
        spec = catalog.effective_specs(item)
        summary = (f"{spec['label']} · {spec['rated_w']:.0f} W · {spec['current_a']:.2f} A" if spec else "no figures yet")
        # The title stays the same while it is edited, so the panel stays open.
        with st.expander(f"{i + 1}. {item['name']}", expanded=st.session_state.get("am_open") == i):
            st.caption(summary)
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            item["usage"] = c1.select_slider("How much it is used", ["Low", "Medium", "High"], value=item.get("usage", "Medium"),
                                             key=f"am_use_{i}")
            if c2.button("Remove", key=f"am_del_{i}", width="stretch", disabled=len(items) <= 1,
                         help="A home needs at least one appliance." if len(items) <= 1 else None):
                items.pop(i)
                _save(db, household_id)
                notify("info", f"{item['name']} removed.")
                st.rerun()
            if spec_editor(item, household_id, f"am_{i}_{item.get('type_key', 'x')}"):
                st.session_state["am_open"] = i
                _save(db, household_id)
                st.rerun()
    if json.dumps(items, sort_keys=True, default=str) != before:
        _save(db, household_id)
