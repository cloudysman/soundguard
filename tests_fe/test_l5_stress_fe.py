"""L5 — stress / performance / leak suite for the SoundGuard frontend.

Goal: hammer the UI (hundreds of slider events, dozens of chip and tab switches, a near-limit
upload) and prove it stays responsive, coherent, and leak-free:
  * no console 'error' entries where every request is a 2xx,
  * no uncaught 'pageerror',
  * no unhandled promise rejections,
  * the DOM does NOT grow unbounded (renderResult must REPLACE innerHTML, not append),
  * a near-limit (~9.5 MB) upload is handled (decode-400 or a result) without hanging.

All selectors verified against static/index.html + soundguard-app.js and DOM_CONTRACT.md.
Debounce for sliders is 80 ms (soundguard-app.js `refreshDebounced`).
"""
import os

import pytest
from playwright.sync_api import expect

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLIDERS = ("noise", "volume", "clip")


# ----------------------------------------------------------------------------- helpers
def _pageerrors(page_errors):
    return [e for e in page_errors if e[0] == "pageerror"]


def _console_errors(page_errors):
    return [e for e in page_errors if e[0] == "console"]


def _node_count(page):
    return page.evaluate("() => document.querySelectorAll('*').length")


def _wait_settled(page):
    """Wait for the debounce + the two /api/predict round-trips to finish."""
    page.wait_for_load_state("networkidle")


# ----------------------------------------------------------------------------- slider storm
def test_slider_storm_stays_responsive_and_coherent(app_page, page_errors):
    """~200 rapid input events across all three sliders. The debounce must coalesce them; the page
    must stay responsive, render a coherent final prediction, and log no console errors."""
    # Fire 200 input events (value 0..100) round-robin across the three cor-sliders, then pin each
    # slider to a known final value so we can assert the labels reflect the last state.
    elapsed = app_page.evaluate(
        """() => {
            const t0 = performance.now();
            const get = k => document.querySelector(
                `input[type=range][data-slider="${k}"][data-group="cor"]`);
            const sliders = ['noise','volume','clip'].map(get);
            for (let i = 0; i < 200; i++){
                const inp = sliders[i % 3];
                inp.value = String(i % 101);
                inp.dispatchEvent(new Event('input', { bubbles: true }));
            }
            // deterministic final state
            const finals = { noise: 40, volume: 30, clip: 20 };
            for (const [k, v] of Object.entries(finals)){
                const inp = get(k);
                inp.value = String(v);
                inp.dispatchEvent(new Event('input', { bubbles: true }));
            }
            return performance.now() - t0;
        }"""
    )
    # The synchronous burst of 203 handler calls must not wedge the main thread.
    assert elapsed < 5000, f"slider storm blocked the main thread for {elapsed:.0f} ms"

    # value labels reflect the final pinned state immediately (syncSliders is synchronous)
    expect(app_page.locator('[data-sval="noise-cor"]')).to_have_text("40%")
    expect(app_page.locator('[data-sval="volume-cor"]')).to_have_text("30%")
    expect(app_page.locator('[data-sval="clip-cor"]')).to_have_text("20%")

    # the debounced refresh eventually renders a coherent prediction
    _wait_settled(app_page)
    app_page.wait_for_selector("#pred-body .top1 .label", timeout=20000)
    label = app_page.inner_text("#pred-body .top1 .label").strip()
    assert label in {SG_pretty(c) for c in CLASSES}, f"incoherent final label {label!r}"
    pct = app_page.inner_text("#pred-body .top1 .pct").strip()
    assert pct.endswith("%")

    assert _pageerrors(page_errors) == [], page_errors
    # every request in this flow is a valid 200 -> no browser console errors
    assert _console_errors(page_errors) == [], _console_errors(page_errors)


# ----------------------------------------------------------------------------- chip storm
def test_chip_switch_storm_consistent(app_page, page_errors):
    """~60 chip switches. No pageerror, exactly one selected chip, and after the flow settles the
    rendered prediction is consistent with the finally-selected chip."""
    chip_ids = ["dog", "siren", "jack", "horn", "kids", "gun", "ac"]
    for i in range(60):
        cid = chip_ids[i % len(chip_ids)]
        app_page.click(f'.chip[data-sample="{cid}"]')

    # exactly one chip pressed, and it is the last one clicked (aria-pressed is set synchronously)
    last = chip_ids[(60 - 1) % len(chip_ids)]
    pressed = app_page.eval_on_selector_all(
        '.chip[data-sample]',
        "els => els.filter(e => e.getAttribute('aria-pressed') === 'true').map(e => e.dataset.sample)",
    )
    assert pressed == [last], f"expected exactly [{last}] pressed, got {pressed}"

    # settle any in-flight racing refreshes, then pin the final selection to 'dog' (known -> dog bark)
    _wait_settled(app_page)
    app_page.click('.chip[data-sample="dog"]')
    _wait_settled(app_page)
    app_page.wait_for_function(
        "() => { const e = document.querySelector('#pred-body .top1 .label');"
        " return e && e.textContent.trim() === 'dog bark'; }",
        timeout=20000,
    )
    assert _pageerrors(page_errors) == [], page_errors
    assert _console_errors(page_errors) == [], _console_errors(page_errors)


# ----------------------------------------------------------------------------- node growth / leak
def test_dom_node_count_bounded_across_heavy_loop(app_page, page_errors):
    """renderResult must REPLACE innerHTML, not append. A heavy interaction loop must not grow the
    total element count unboundedly."""
    _wait_settled(app_page)
    before = _node_count(app_page)

    chip_ids = ["dog", "siren", "jack", "horn", "kids", "gun", "ac"]
    for i in range(40):
        app_page.click(f'.chip[data-sample="{chip_ids[i % len(chip_ids)]}"]')
    # a batch of slider churn on top of the chip churn
    app_page.evaluate(
        """() => {
            const get = k => document.querySelector(
                `input[type=range][data-slider="${k}"][data-group="cor"]`);
            for (let i = 0; i < 60; i++){
                const inp = get(['noise','volume','clip'][i % 3]);
                inp.value = String(i % 101);
                inp.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }"""
    )
    _wait_settled(app_page)
    app_page.wait_for_selector("#pred-body .top1", timeout=20000)
    after = _node_count(app_page)

    growth = after - before
    # innerHTML replacement keeps the count essentially flat; allow a tiny slack for status-card
    # class churn. A leak (append instead of replace) would blow far past this after 100 renders.
    assert growth <= 40, (
        f"DOM node count grew by {growth} (before={before}, after={after}); "
        "renderResult may be appending instead of replacing"
    )
    assert _pageerrors(page_errors) == [], page_errors


# ----------------------------------------------------------------------------- tab storm
def test_tab_switch_storm(app_page, page_errors):
    """Rapid tab switching x50. No pageerror; exactly one active panel; final active panel matches
    the last-clicked tab; node count stays bounded."""
    before = _node_count(app_page)
    tabs = ["inference", "corruption", "dashboard"]
    # end deterministically on 'inference' (50 % 3 sequence ends on index 49 % 3 == 1 -> corruption),
    # so append one explicit final click to inference for a known end-state.
    app_page.evaluate(
        """() => {
            const order = ['inference','corruption','dashboard'];
            for (let i = 0; i < 50; i++){
                document.querySelector(`.tab[data-tab="${order[i % 3]}"]`).click();
            }
            document.querySelector('.tab[data-tab="inference"]').click();
        }"""
    )
    active = app_page.eval_on_selector_all(
        ".tab-panel.active", "els => els.map(e => e.id)"
    )
    assert active == ["tab-inference"], f"expected only tab-inference active, got {active}"
    selected = app_page.eval_on_selector_all(
        ".tab",
        "els => els.filter(e => e.getAttribute('aria-selected') === 'true').map(e => e.dataset.tab)",
    )
    assert selected == ["inference"], f"expected only inference selected, got {selected}"

    after = _node_count(app_page)
    assert after - before <= 40, f"tab storm grew DOM by {after - before} nodes"
    assert _pageerrors(page_errors) == [], page_errors
    assert _console_errors(page_errors) == [], _console_errors(page_errors)


# ----------------------------------------------------------------------------- unhandled rejections
def test_no_unhandled_promise_rejections(page, base_url, page_errors):
    """Hook window 'unhandledrejection' BEFORE load, then run a mixed heavy loop; the app's
    try/catch around every fetch must leave zero unhandled rejections."""
    page.add_init_script(
        "window.__rejections = [];"
        "window.addEventListener('unhandledrejection',"
        " e => window.__rejections.push(String(e.reason)));"
    )
    page.goto(base_url + "/")
    page.wait_for_selector("#pred-body .top1", timeout=20000)

    chip_ids = ["dog", "siren", "jack", "horn", "kids", "gun", "ac"]
    for i in range(20):
        page.click(f'.chip[data-sample="{chip_ids[i % len(chip_ids)]}"]')
    page.evaluate(
        """() => {
            const get = k => document.querySelector(
                `input[type=range][data-slider="${k}"][data-group="cor"]`);
            for (let i = 0; i < 60; i++){
                const inp = get(['noise','volume','clip'][i % 3]);
                inp.value = String(i % 101);
                inp.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }"""
    )
    page.wait_for_load_state("networkidle")
    rejections = page.evaluate("() => window.__rejections")
    assert rejections == [], f"unhandled promise rejections: {rejections}"
    assert _pageerrors(page_errors) == [], page_errors


# ----------------------------------------------------------------------------- near-limit upload
def test_near_limit_upload_is_handled_without_hanging(app_page, page_errors):
    """A ~9.5 MB audio/wav upload (under the 10 MB ceiling) must be handled — a decode 400 surfaced
    as a Backend-error status card, or a real result — without hanging the UI."""
    size = int(9.5 * 1024 * 1024)  # ~9.96e6 bytes, below MAX_UPLOAD_BYTES (10 MB)
    buf = b"\x00" * size

    # Wrap the upload in an explicit response wait — the anti-hang guard is that the POST actually
    # RESOLVES with a definitive HTTP status within the timeout (never hangs on a ~9.5 MB body).
    with app_page.expect_response("**/api/predict/upload", timeout=25000) as resp_info:
        app_page.set_input_files(
            "#file-input",
            files=[{"name": "near_limit.wav", "mimeType": "audio/wav", "buffer": buf}],
        )
    resp = resp_info.value
    assert resp.status in (200, 400, 413, 415), f"unexpected upload status {resp.status}"

    if resp.status == 200:
        # Decoder accepted it — a fresh prediction must render.
        app_page.wait_for_selector("#pred-body .top1 .label", timeout=20000)
        assert app_page.inner_text("#pred-body .top1 .label").strip() != ""
    else:
        # Non-2xx (expected: 400 decode) -> showError paints every status card as Backend error.
        app_page.wait_for_function(
            "() => [...document.querySelectorAll('[data-status-card] .status-name')]"
            ".some(n => n.textContent.trim() === 'Backend error')",
            timeout=20000,
        )
        names = app_page.eval_on_selector_all(
            "[data-status-card] .status-name", "els => els.map(e => e.textContent.trim())"
        )
        assert names == ["Backend error", "Backend error"], names
        reason = app_page.inner_text("[data-status-card] .status-reason")
        assert reason.strip() != ""
    # no uncaught pageerror (the non-2xx is caught by refreshUpload's try/catch)
    assert _pageerrors(page_errors) == [], page_errors


# ----------------------------------------------------------------------------- module constants
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


def SG_pretty(s):
    return s.replace("_", " ")
