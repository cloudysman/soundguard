"""L2 — user interactions for the SoundGuard frontend.

Covers real interactions: sliders (label + fresh POST + qmeters + conf-compare + SNR + status),
canvas existence/redraw, reset-to-clean, tab switching (aria/active/preserve), confidence marker
tracking, now-playing on chip pick, real WAV upload, and keyboard reachability/activation.

Fixtures (from conftest.py): app_page, page, base_url, page_errors. Do NOT redefine them.
Selectors follow tests_fe/DOM_CONTRACT.md and were verified against the live DOM.
"""
import os
import re

import pytest
from playwright.sync_api import expect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOG_WAV = os.path.join(ROOT, "samples", "dog.wav")


# ---------- helpers ----------
def _is_predict(req):
    """A POST to /api/predict (NOT /api/predict/upload)."""
    return req.method == "POST" and req.url.rstrip("/").endswith("/api/predict")


def _num(text):
    """First (possibly signed/decimal) number in a string, e.g. '-1 dB' -> -1.0."""
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    assert m, f"no number in {text!r}"
    return float(m.group())


def go_corruption(page):
    page.click('.tab[data-tab="corruption"]')
    page.wait_for_selector("#tab-corruption.active")


def slider(page, which):
    return page.locator(f'input[data-slider="{which}"][data-group="cor"]')


def set_slider(page, which, value):
    """Set a range slider to `value` (fires the app's 'input' handler)."""
    slider(page, which).fill(str(value))


def snr_meter_value(page):
    """The SNR quality meter's value locator (SNR is the first qmeter)."""
    first = page.locator("#qmeters .qmeter").nth(0)
    return first.locator(".qv")


def no_pageerror(errs):
    return [e for e in errs if e[0] == "pageerror"]


# ---------- sliders: label + fresh prediction ----------
@pytest.mark.parametrize("which,value", [("noise", 40), ("volume", 55), ("clip", 30)])
def test_slider_updates_label_and_fires_predict(app_page, which, value):
    go_corruption(app_page)
    with app_page.expect_request(_is_predict):
        set_slider(app_page, which, value)
    expect(app_page.locator(f'[data-sval="{which}-cor"]')).to_have_text(f"{value}%")


def test_slider_refreshes_qmeters_and_compare(app_page):
    go_corruption(app_page)
    snr = snr_meter_value(app_page)
    base_snr = snr.inner_text()
    corr_big = app_page.locator('#conf-compare .cc-cell:not(.clean) .big')
    base_big = corr_big.inner_text()
    clean_big = app_page.locator('#conf-compare .cc-cell.clean .big')
    base_clean = clean_big.inner_text()

    set_slider(app_page, "noise", 85)

    # qmeters re-rendered (SNR value moved) and the corrupted compare cell moved
    expect(snr).not_to_have_text(base_snr)
    expect(corr_big).not_to_have_text(base_big)
    # the clean baseline cell stays put (same deterministic clean input)
    expect(clean_big).to_have_text(base_clean)
    # a drop is reported in points
    expect(app_page.locator("#conf-compare")).to_contain_text("pts")


def test_noise_lowers_snr_qmeter(app_page):
    go_corruption(app_page)
    # confirm the first qmeter is really SNR
    expect(app_page.locator("#qmeters .qmeter").nth(0).locator(".qk")).to_have_text("SNR")
    qv = snr_meter_value(app_page)
    base_text = qv.inner_text()
    base = _num(base_text)

    set_slider(app_page, "noise", 80)
    expect(qv).not_to_have_text(base_text)
    after = _num(qv.inner_text())
    assert after < base, f"SNR should drop with noise: {base} -> {after}"


def test_noise_pushes_status_toward_degraded(app_page):
    go_corruption(app_page)
    card = app_page.locator('#tab-corruption [data-status-card]')
    with app_page.expect_request(_is_predict):
        set_slider(app_page, "noise", 95)
    expect(card).to_have_class(re.compile(r"\b(uncertain|reject)\b"))


# ---------- reset-to-clean ----------
def test_reset_returns_sliders_and_status_to_clean(app_page):
    go_corruption(app_page)
    card = app_page.locator('#tab-corruption [data-status-card]')
    clean_class = card.get_attribute("class")
    clean_status = clean_class.split()[-1]  # e.g. 'reliable'

    # degrade first
    set_slider(app_page, "noise", 90)
    expect(card).to_have_class(re.compile(r"\b(uncertain|reject)\b"))

    # reset
    with app_page.expect_request(_is_predict):
        app_page.click("[data-reset]")

    for w in ("noise", "volume", "clip"):
        expect(app_page.locator(f'[data-sval="{w}-cor"]')).to_have_text("0%")
        assert slider(app_page, w).input_value() == "0"
    # status walks back to the original clean verdict
    expect(card).to_have_class(re.compile(rf"\b{clean_status}\b"))


# ---------- canvases ----------
def test_canvases_exist_with_declared_dimensions(app_page):
    for cid, w, h in [("wave-inf", 760, 230), ("mel-inf", 760, 230),
                      ("wave-cor", 760, 220), ("mel-cor", 760, 220)]:
        c = app_page.locator(f"#{cid}")
        expect(c).to_have_count(1)
        assert c.get_attribute("width") == str(w), cid
        assert c.get_attribute("height") == str(h), cid


def test_corruption_canvas_is_redrawn(app_page):
    go_corruption(app_page)
    before = app_page.eval_on_selector("#wave-cor", "c => c.toDataURL()")
    qv = snr_meter_value(app_page)
    base = qv.inner_text()
    set_slider(app_page, "noise", 80)
    expect(qv).not_to_have_text(base)  # render has landed
    after = app_page.eval_on_selector("#wave-cor", "c => c.toDataURL()")
    assert before != after, "waveform canvas should be redrawn after corruption"


# ---------- tabs ----------
@pytest.mark.parametrize("tab", ["inference", "corruption", "dashboard"])
def test_tab_switch_toggles_aria_and_active(app_page, tab):
    app_page.click(f'.tab[data-tab="{tab}"]')
    expect(app_page.locator(f'.tab[data-tab="{tab}"]')).to_have_attribute("aria-selected", "true")
    expect(app_page.locator(f"#tab-{tab}")).to_have_class(re.compile(r"\bactive\b"))
    for other in ("inference", "corruption", "dashboard"):
        if other == tab:
            continue
        expect(app_page.locator(f'.tab[data-tab="{other}"]')).to_have_attribute("aria-selected", "false")
        assert "active" not in (app_page.locator(f"#tab-{other}").get_attribute("class") or "")


def test_tab_switch_preserves_prediction(app_page):
    label = app_page.inner_text("#pred-body .top1 .label")
    pct = app_page.inner_text("#pred-body .top1 .pct")
    app_page.click('.tab[data-tab="corruption"]')
    app_page.wait_for_selector("#tab-corruption.active")
    app_page.click('.tab[data-tab="inference"]')
    app_page.wait_for_selector("#tab-inference.active")
    expect(app_page.locator("#pred-body .top1 .label")).to_have_text(label)
    expect(app_page.locator("#pred-body .top1 .pct")).to_have_text(pct)


# ---------- confidence marker ----------
def test_confidence_marker_tracks_confidence(app_page):
    conf = _num(app_page.inner_text("#conf-body .conf-num"))  # e.g. "91%"
    left = app_page.eval_on_selector("#conf-body .marker", "m => m.style.left")
    assert left.endswith("%"), left
    left_val = float(left[:-1])
    assert abs(left_val - conf) < 1.5, f"marker left {left_val} vs conf {conf}"


# ---------- now-playing on chip pick ----------
@pytest.mark.parametrize("sample,expname", [
    ("siren", "siren.wav"),
    ("gun", "gun_shot.wav"),
    ("ac", "air_conditioner.wav"),
])
def test_chip_pick_updates_nowplaying_and_predicts(app_page, sample, expname):
    with app_page.expect_request(_is_predict):
        app_page.click(f'.chip[data-sample="{sample}"]')
    expect(app_page.locator("[data-nowplaying]").first).to_contain_text(expname)
    expect(app_page.locator(f'.chip[data-sample="{sample}"]')).to_have_attribute("aria-pressed", "true")
    # the previously-default dog chip is no longer pressed
    expect(app_page.locator('.chip[data-sample="dog"]')).to_have_attribute("aria-pressed", "false")


# ---------- real WAV upload ----------
def test_upload_real_wav_predicts_dog_and_sets_nowplaying(app_page):
    with app_page.expect_response(lambda r: "/api/predict/upload" in r.url) as resp:
        app_page.set_input_files("#file-input", DOG_WAV)
    assert resp.value.ok, f"upload responded {resp.value.status}"
    expect(app_page.locator("[data-nowplaying]").first).to_contain_text("dog.wav")
    expect(app_page.locator("#pred-body .top1 .label")).to_have_text("dog bark")


# ---------- keyboard reachability / activation ----------
@pytest.mark.parametrize("key", ["Enter", "Space"])
def test_keyboard_activates_tab(app_page, key):
    tab = app_page.locator('.tab[data-tab="dashboard"]')
    tab.focus()
    assert tab.evaluate("el => el === document.activeElement"), "tab must be keyboard-focusable"
    tab.press(key)
    expect(tab).to_have_attribute("aria-selected", "true")
    expect(app_page.locator("#tab-dashboard")).to_have_class(re.compile(r"\bactive\b"))


@pytest.mark.parametrize("key", ["Enter", "Space"])
def test_keyboard_activates_chip(app_page, key):
    chip = app_page.locator('.chip[data-sample="siren"]')
    chip.focus()
    assert chip.evaluate("el => el === document.activeElement"), "chip must be keyboard-focusable"
    with app_page.expect_request(_is_predict):
        chip.press(key)
    expect(chip).to_have_attribute("aria-pressed", "true")
    expect(app_page.locator("[data-nowplaying]").first).to_contain_text("siren.wav")


# ---------- no uncaught errors across a full interaction sequence ----------
def test_no_pageerror_during_interactions(app_page, page_errors):
    go_corruption(app_page)
    set_slider(app_page, "noise", 50)
    set_slider(app_page, "clip", 40)
    with app_page.expect_request(_is_predict):
        app_page.click("[data-reset]")
    app_page.click('.tab[data-tab="dashboard"]')
    app_page.wait_for_selector("#tab-dashboard.active")
    app_page.click('.tab[data-tab="inference"]')
    with app_page.expect_request(_is_predict):
        app_page.click('.chip[data-sample="gun"]')
    assert no_pageerror(page_errors) == [], page_errors
