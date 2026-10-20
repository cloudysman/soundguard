"""L3 — error handling / fault injection for the SoundGuard frontend.

Two kinds of faults are exercised:

  1. Route-injected backend faults on /api/predict and /api/dashboard (page.route → fulfil/abort).
     The app must surface an *error state* (the "Backend error" status card / chart-empty message),
     never a fabricated prediction, and must not raise an uncaught pageerror.

  2. Genuine upload validation through the REAL backend (no route): a .txt (415), an empty file (400),
     an oversized >10 MB file (413), an undecodable "audio/wav" blob (400) and a WebM recording blob
     (400 with the WAV hint). Each 4xx `detail` must reach the status card verbatim.

The distinguishing signal for "error state, not a fabricated prediction" is that `showError` sets
`.status-name` to exactly "Backend error" and the card class to `status-card reject` — a genuine
prediction would instead render a real status name ("Reliable"/"Uncertain"/"Reject") via renderStatus.
"""
import os

import pytest
from playwright.sync_api import expect

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- helpers
def _pageerrors(page_errors):
    return [e for e in page_errors if e[0] == "pageerror"]


def _assert_backend_error_card(page, reason_substr=None):
    """The inference status card must show the injected/real error, not a fabricated prediction."""
    card = page.locator("[data-status-card]").first
    name = card.locator(".status-name")
    expect(name).to_have_text("Backend error", timeout=10000)
    # class flips to the reject styling used by showError
    expect(card).to_have_class("card status-card reject")
    # tag is the fixed showError tag (proves it is showError, not renderStatus)
    expect(card.locator(".status-tag")).to_have_text("request failed")
    if reason_substr is not None:
        expect(card.locator(".status-reason")).to_contain_text(reason_substr, timeout=10000)


def _route_predict(page, handler):
    page.route("**/api/predict", handler)


def _trigger_predict(page):
    """Click a non-default sample chip to force a fresh /api/predict refresh."""
    page.click('.chip[data-sample="siren"]')


def _do_upload(page, name, mime, buf):
    page.set_input_files("#file-input", files=[{"name": name, "mimeType": mime, "buffer": buf}])


# =========================================================================== #
#  1. /api/predict route-injected faults                                      #
# =========================================================================== #
@pytest.mark.parametrize(
    "status,body,ctype,reason",
    [
        (503, '{"detail":"Model unavailable — run: pip install -r requirements.txt"}',
         "application/json", "Model unavailable"),
        (422, '{"detail":"corr.noise: input should be less than or equal to 1"}',
         "application/json", "input should be less than or equal to 1"),
        (500, '{"detail":"internal server error"}',
         "application/json", "internal server error"),
    ],
)
def test_predict_error_status_shows_detail(app_page, page_errors, status, body, ctype, reason):
    """503 / 422 / 500 with a JSON {detail} → Backend error card carrying the detail text."""
    _route_predict(app_page, lambda route: route.fulfill(status=status, content_type=ctype, body=body))
    _trigger_predict(app_page)
    _assert_backend_error_card(app_page, reason_substr=reason)
    assert _pageerrors(page_errors) == [], page_errors


def test_predict_500_without_json_uses_fallback_message(app_page, page_errors):
    """A non-2xx whose body is not JSON → fetchPredict's `.catch(()=>({}))` path → fallback message
    'predict failed (500)' still surfaces as an error state (no crash on the bad body)."""
    _route_predict(app_page, lambda route: route.fulfill(
        status=500, content_type="text/plain", body="upstream exploded"))
    _trigger_predict(app_page)
    _assert_backend_error_card(app_page, reason_substr="predict failed (500)")
    assert _pageerrors(page_errors) == [], page_errors


def test_predict_network_abort_surfaces_error(app_page, page_errors):
    """A hard network failure (route.abort) rejects fetch → showError, no unhandled rejection."""
    _route_predict(app_page, lambda route: route.abort())
    _trigger_predict(app_page)
    # message text is browser-defined ("Failed to fetch"); only require the error state.
    _assert_backend_error_card(app_page)
    # the reject card carries a non-empty reason
    reason = app_page.locator("[data-status-card] .status-reason").first
    expect(reason).not_to_have_text("")
    assert _pageerrors(page_errors) == [], page_errors


def test_predict_fault_does_not_fabricate_a_prediction(app_page, page_errors):
    """After a predict fault, the status card must NOT read as a genuine verdict."""
    _route_predict(app_page, lambda route: route.fulfill(
        status=503, content_type="application/json", body='{"detail":"down"}'))
    _trigger_predict(app_page)
    _assert_backend_error_card(app_page, reason_substr="down")
    # genuine renderStatus would produce one of these names; showError must not.
    name = app_page.locator("[data-status-card] .status-name").first.inner_text().strip()
    assert name not in ("Reliable", "Uncertain", "Reject"), name
    assert _pageerrors(page_errors) == [], page_errors


# =========================================================================== #
#  2. /api/dashboard route-injected fault                                     #
# =========================================================================== #
def test_dashboard_503_shows_chart_empty_in_all_four_containers(page, base_url, page_errors):
    """Must be routed BEFORE navigation because loadDashboard() runs on init.
    A 503 → showChartsUnavailable() writes a .chart-empty (with a <code>) into all four containers."""
    page.route("**/api/dashboard",
               lambda route: route.fulfill(status=503, content_type="application/json",
                                           body='{"detail":"No evaluation results — run eval.py."}'))
    page.goto(base_url + "/")
    # predict still works, so the page reaches its normal rendered state
    page.wait_for_selector("#pred-body .top1", timeout=20000)

    containers = ["#chart-reliability", "#chart-risk", "#metrics-table", "#classacc-rows"]
    for sel in containers:
        empty = page.locator(f"{sel} .chart-empty")
        expect(empty).to_have_count(1)
        # the message embeds the remediation command inside a <code>
        expect(page.locator(f"{sel} .chart-empty code")).to_have_count(1)
        expect(empty).to_contain_text("No evaluation results")
    # no SVG charts were fabricated from missing data
    assert page.locator("#chart-reliability svg").count() == 0
    assert page.locator("#metrics-table table").count() == 0
    assert _pageerrors(page_errors) == [], page_errors


# =========================================================================== #
#  3. Genuine upload validation through the REAL backend (no route)           #
# =========================================================================== #
def test_upload_txt_file_415(app_page, page_errors):
    """Non-audio content-type + non-audio extension → 415 'Unsupported file type'."""
    _do_upload(app_page, "notes.txt", "text/plain", b"just some plain text, definitely not audio")
    _assert_backend_error_card(app_page, reason_substr="Unsupported file type")
    assert _pageerrors(page_errors) == [], page_errors


def test_upload_empty_file_400(app_page, page_errors):
    """A zero-byte audio file → 400 'Empty file.' (size check before decode)."""
    _do_upload(app_page, "empty.wav", "audio/wav", b"")
    _assert_backend_error_card(app_page, reason_substr="Empty file")
    assert _pageerrors(page_errors) == [], page_errors


def test_upload_oversized_file_413(app_page, page_errors):
    """A >10 MB audio upload → 413 'File too large' (rejected before the model runs)."""
    big = b"RIFF" + b"\x00" * (10 * 1024 * 1024 + 100)   # just over the 10 MB ceiling
    _do_upload(app_page, "huge.wav", "audio/wav", big)
    _assert_backend_error_card(app_page, reason_substr="File too large")
    assert _pageerrors(page_errors) == [], page_errors


def test_upload_undecodable_wav_blob_400(app_page, page_errors):
    """Bytes that claim audio/wav but are not decodable → 400 'Could not decode audio'."""
    _do_upload(app_page, "corrupt.wav", "audio/wav", b"this is not a real WAV payload at all")
    _assert_backend_error_card(app_page, reason_substr="Could not decode audio")
    assert _pageerrors(page_errors) == [], page_errors


def test_upload_webm_recording_blob_400_with_wav_hint(app_page, page_errors):
    """Simulates the browser-records-WebM case: audio/webm blob → 400 whose detail hints WAV."""
    # a plausible WebM/EBML magic header + junk; soundfile cannot decode it
    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
    _do_upload(app_page, "mic_recording.webm", "audio/webm", webm)
    _assert_backend_error_card(app_page, reason_substr="WAV")
    # and the fuller message is present
    expect(app_page.locator("[data-status-card] .status-reason").first).to_contain_text(
        "Could not decode audio")
    assert _pageerrors(page_errors) == [], page_errors
