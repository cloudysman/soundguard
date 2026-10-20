"""L5 — client-side injection / security surface of the SoundGuard frontend.

The app builds DOM with template-literal `innerHTML` in two places that carry attacker-influenced
strings:
  * nowPlaying(name, extra)  -> [data-nowplaying].innerHTML = `<b>${esc(name)}</b> · ${extra}`   (name = uploaded filename)
  * showError(msg)           -> [data-status-card].innerHTML = `...<div class="status-reason">${esc(msg)}</div>...`
                               (msg = backend JSON `detail` on a non-2xx response)

Both untrusted strings (a filename, a server error `detail`) are now routed through esc(), which
HTML-escapes &<>"' before they reach innerHTML. An injected `<img src=x onerror=...>` payload is
therefore neutralised: it becomes inert text, the onerror handler never runs, and no element is
created. These tests assert that neutralised behaviour — the payload flags stay undefined, no <img>
is injected, and the literal payload text is shown escaped.

Benign inputs (normal filenames, normal status class/text) are also asserted to be safe.
"""
import json
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOG_WAV = os.path.join(PROJECT_ROOT, "samples", "dog.wav")

FILENAME_PAYLOAD = '<img src=x onerror="window.__xss=1">.wav'
DETAIL_PAYLOAD = '<img src=x onerror="window.__xss2=1">'


def _dog_bytes():
    with open(DOG_WAV, "rb") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# (1) FILENAME XSS — unsanitised filename -> nowPlaying innerHTML
# --------------------------------------------------------------------------- #
def test_filename_xss_neutralised_via_nowplaying(page, base_url):
    """Uploading a file whose NAME is an <img onerror> payload MUST NOT execute: nowPlaying()
    escapes the filename via esc() before innerHTML, so the payload becomes inert text — no
    onerror fires (window.__xss stays undefined), no <img> is injected, and the literal filename
    is shown as text inside the <b> label."""
    page.goto(base_url + "/")
    page.wait_for_selector("#pred-body .top1", timeout=20000)
    # deliberately do NOT initialise window.__xss: it must stay undefined if the payload is inert.

    # Upload real WAV bytes under a malicious filename. nowPlaying(f.name, 'uploaded') runs
    # synchronously on the change event, before/independent of the upload round-trip.
    page.set_input_files(
        "#file-input",
        files=[{"name": FILENAME_PAYLOAD, "mimeType": "audio/wav", "buffer": _dog_bytes()}],
    )

    # Deterministic sync point: the escaped filename lands as literal TEXT in the <b> label.
    page.wait_for_function(
        "(name) => Array.from(document.querySelectorAll('[data-nowplaying] b'))"
        ".some(b => b.textContent === name)",
        arg=FILENAME_PAYLOAD,
        timeout=6000,
    )

    executed = page.evaluate("() => window.__xss")
    injected_imgs = page.locator("[data-nowplaying] img").count()
    shown_text = page.locator("[data-nowplaying] b").first.inner_text()

    # SECURE expectation (now satisfied by the fixed app):
    assert executed is None, "filename XSS executed: onerror handler ran (window.__xss set)"
    assert injected_imgs == 0, "filename XSS: <img> element injected into [data-nowplaying]"
    assert shown_text == FILENAME_PAYLOAD, "filename not rendered literally as escaped text"


# --------------------------------------------------------------------------- #
# (2) BACKEND-DETAIL HTML INJECTION — unsanitised `detail` -> showError innerHTML
# --------------------------------------------------------------------------- #
def test_backend_detail_html_injection_neutralised_via_showerror(page, base_url):
    """A 503 from /api/predict whose JSON `detail` contains an <img onerror> payload MUST NOT
    execute: showError() escapes the detail via esc() before innerHTML, so the payload becomes
    inert text — no onerror fires (window.__xss2 stays undefined), no <img> is injected into
    .status-reason, and the literal detail string is shown as text."""
    # json.dumps so the payload's own double quotes are properly escaped -> VALID JSON, so the
    # app's `res.json()` parse succeeds and e.detail reaches showError (a malformed body would be
    # swallowed by `.catch(()=>({}))` and the payload would never render).
    body = json.dumps({"detail": DETAIL_PAYLOAD})

    def handler(route):
        route.fulfill(status=503, content_type="application/json", body=body)

    page.route("**/api/predict", handler)
    # deliberately do NOT initialise window.__xss2: it must stay undefined if the payload is inert.
    page.goto(base_url + "/")

    # The initial refresh() POSTs /api/predict -> 503 -> showError(detail). Deterministic sync
    # point: the escaped detail lands as literal TEXT in .status-reason.
    page.wait_for_function(
        "(payload) => Array.from(document.querySelectorAll('[data-status-card] .status-reason'))"
        ".some(el => el.textContent.includes(payload))",
        arg=DETAIL_PAYLOAD,
        timeout=8000,
    )

    executed = page.evaluate("() => window.__xss2")
    injected_imgs = page.locator("[data-status-card] .status-reason img").count()
    shown_text = page.locator("[data-status-card] .status-reason").first.inner_text()

    # SECURE expectation (now satisfied by the fixed app):
    assert executed is None, "backend-detail XSS executed: onerror handler ran (window.__xss2 set)"
    assert injected_imgs == 0, "backend-detail XSS: <img> injected into .status-reason"
    assert DETAIL_PAYLOAD in shown_text, "detail not rendered literally as escaped text"


# --------------------------------------------------------------------------- #
# (3) Confirm benign inputs are NOT a vector (secure behaviour — these pass)
# --------------------------------------------------------------------------- #
def test_benign_filename_rendered_as_text_no_injection(page, base_url):
    """A normal filename must render as plain text: no <img>/<script> injected, no globals set."""
    page.goto(base_url + "/")
    page.wait_for_selector("#pred-body .top1", timeout=20000)

    page.set_input_files(
        "#file-input",
        files=[{"name": "harmless_clip.wav", "mimeType": "audio/wav", "buffer": _dog_bytes()}],
    )
    # nowPlaying renders synchronously; wait for the visible <b>filename</b> to appear.
    page.wait_for_function(
        "() => Array.from(document.querySelectorAll('[data-nowplaying] b'))"
        ".some(b => b.textContent === 'harmless_clip.wav')",
        timeout=6000,
    )

    assert page.locator("[data-nowplaying] img").count() == 0
    assert page.locator("[data-nowplaying] script").count() == 0
    # a benign filename never trips the payload flags
    assert not page.evaluate("() => window.__xss === 1")


def test_benign_backend_detail_rendered_as_text_no_injection(page, base_url):
    """A non-HTML backend `detail` shows up as the status reason text — no element injection."""
    detail = "prediction failed: model unavailable"
    body = '{"detail": "%s"}' % detail

    def handler(route):
        route.fulfill(status=503, content_type="application/json", body=body)

    page.route("**/api/predict", handler)
    page.goto(base_url + "/")

    # showError overwrites every status card; wait for our detail text to land in .status-reason.
    page.wait_for_function(
        "(d) => Array.from(document.querySelectorAll('[data-status-card] .status-reason'))"
        ".some(el => el.textContent.includes(d))",
        arg=detail,
        timeout=8000,
    )

    assert page.locator("[data-status-card] .status-reason img").count() == 0
    assert page.locator("[data-status-card] .status-reason script").count() == 0
    # "Backend error" header confirms we went through showError, not a normal render
    assert page.locator("[data-status-card] .status-name").first.inner_text().strip() == "Backend error"


def test_normal_status_class_and_text_not_a_vector(app_page, page_errors):
    """The happy-path status render uses only known class names / static text and injects no
    active content. Confirms normal rendering is not itself an injection sink."""
    card = app_page.locator("[data-status-card]").first
    cls = card.get_attribute("class")
    # className is composed from a fixed status literal (reliable|uncertain|reject) — no attacker input
    assert cls is not None
    tokens = set(cls.split())
    assert "status-card" in tokens
    assert tokens & {"reliable", "uncertain", "reject"}, cls

    # no active/injected content anywhere in the status cards or now-playing on the happy path
    assert app_page.locator("[data-status-card] img").count() == 0
    assert app_page.locator("[data-status-card] script").count() == 0
    assert app_page.locator("[data-nowplaying] img").count() == 0

    # the status-name is one of the three known human labels (not HTML)
    name = app_page.locator("[data-status-card] .status-name").first.inner_text().strip()
    assert name in {"Reliable", "Uncertain", "Reject"}

    # a benign page must not raise uncaught errors
    assert [e for e in page_errors if e[0] == "pageerror"] == [], page_errors


# --------------------------------------------------------------------------- #
# (4) No eval / no obvious open-redirect sink in the shipped client JS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("asset", ["/soundguard-app.js", "/soundguard-data.js"])
def test_no_eval_or_open_redirect_in_client_js(app_page, base_url, asset):
    """Static safety net: the shipped JS must not use eval()/new Function() or perform an
    attacker-controllable navigation (location assignment / window.open)."""
    resp = app_page.request.get(base_url + asset)
    assert resp.ok, f"{asset} -> {resp.status}"
    src = resp.text()

    assert "eval(" not in src, "eval() present in client JS"
    assert "new Function(" not in src, "new Function() present in client JS"
    # navigation sinks that could enable open-redirect / javascript: URLs
    assert "window.open(" not in src, "window.open() present in client JS"
    assert "location.href" not in src, "location.href assignment present in client JS"
    assert "location.replace" not in src, "location.replace present in client JS"
    assert "location.assign" not in src, "location.assign present in client JS"
