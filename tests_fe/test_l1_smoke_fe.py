"""L1 smoke / structure tests for the SoundGuard frontend.

Happy-path structure only: page shell, tabs, sample chips, input controls, sliders,
default prediction, per-chip prediction render, dashboard charts/tables, corruption-lab
widgets. Selectors verified against static/index.html + soundguard-app.js and DOM_CONTRACT.md.
"""
import pytest
from playwright.sync_api import expect


# data-sample -> visible chip label (from static/index.html)
CHIPS = [
    ("dog", "dog_bark"),
    ("siren", "siren"),
    ("jack", "jackhammer"),
    ("horn", "car_horn"),
    ("kids", "children_playing"),
    ("gun", "gun_shot"),
    ("ac", "air_conditioner"),
]
STATUS_NAMES = {"Reliable", "Uncertain", "Reject"}


def _no_pageerror(page_errors):
    return [e for e in page_errors if e[0] == "pageerror"]


# ---------------- page shell ----------------

def test_page_title(app_page):
    assert app_page.title() == "SoundGuard — Environmental Sound Classification"


def test_header_brand_present(app_page):
    expect(app_page.locator("header.app-header .brand-name")).to_have_text("SoundGuard")
    expect(app_page.locator("header.app-header .brand-sub")).to_be_visible()


def test_exactly_three_tabs_with_correct_data_tab(app_page):
    tabs = app_page.locator(".tab")
    expect(tabs).to_have_count(3)
    assert tabs.evaluate_all("els => els.map(e => e.dataset.tab)") == [
        "inference", "corruption", "dashboard"
    ]


def test_default_selected_tab_is_inference(app_page):
    expect(app_page.locator('.tab[data-tab="inference"]')).to_have_attribute("aria-selected", "true")
    expect(app_page.locator("#tab-inference")).to_have_class("tab-panel active")


# ---------------- sample chips ----------------

def test_seven_sample_chips_present(app_page):
    expect(app_page.locator(".chip[data-sample]")).to_have_count(7)


@pytest.mark.parametrize("data_sample,label", CHIPS)
def test_chip_value_and_label(app_page, data_sample, label):
    chip = app_page.locator(f'.chip[data-sample="{data_sample}"]')
    expect(chip).to_have_count(1)
    expect(chip).to_have_text(label)


# ---------------- input controls ----------------

def test_upload_and_record_buttons_and_hidden_file_input(app_page):
    expect(app_page.locator("#btn-upload")).to_be_visible()
    expect(app_page.locator("#btn-record")).to_be_visible()
    expect(app_page.locator("#btn-record .lbl")).to_have_text("Record")
    file_input = app_page.locator("#file-input")
    expect(file_input).to_have_count(1)
    expect(file_input).to_be_hidden()
    expect(file_input).to_have_attribute("accept", "audio/*")


# ---------------- corruption sliders ----------------

@pytest.mark.parametrize("slider", ["noise", "volume", "clip"])
def test_corruption_slider_exists_starting_at_zero(app_page, slider):
    inp = app_page.locator(f'input[type=range][data-slider="{slider}"][data-group="cor"]')
    expect(inp).to_have_count(1)
    expect(inp).to_have_attribute("min", "0")
    expect(inp).to_have_attribute("max", "100")
    expect(inp).to_have_js_property("value", "0")


def test_slider_value_labels_start_at_zero(app_page):
    for slider in ["noise", "volume", "clip"]:
        expect(app_page.locator(f'[data-sval="{slider}-cor"]')).to_have_text("0%")


# ---------------- default prediction ----------------

def test_default_chip_is_dog_and_aria_pressed(app_page):
    expect(app_page.locator('.chip[data-sample="dog"]')).to_have_attribute("aria-pressed", "true")
    # exactly one chip is pressed
    pressed = app_page.locator('.chip[data-sample][aria-pressed="true"]')
    expect(pressed).to_have_count(1)


def test_default_prediction_top_class_is_dog_bark(app_page):
    expect(app_page.locator("#pred-body .top1 .label")).to_have_text("dog bark")


# ---------------- per-chip prediction render ----------------

@pytest.mark.parametrize("data_sample,label", CHIPS)
def test_clicking_chip_renders_prediction(app_page, data_sample, label):
    chip = app_page.locator(f'.chip[data-sample="{data_sample}"]')
    with app_page.expect_response(lambda r: "/api/predict" in r.url and r.request.method == "POST"):
        chip.click()
    expect(chip).to_have_attribute("aria-pressed", "true")

    # top-1 label present + non-empty
    top1 = app_page.locator("#pred-body .top1 .label")
    expect(top1).to_be_visible()
    assert top1.inner_text().strip() != ""

    # exactly 3 prediction bars, first is the lead
    bars = app_page.locator("#pred-body .bars .bar-row")
    expect(bars).to_have_count(3)
    expect(bars.first).to_have_class("bar-row lead")
    # each bar has name / fill / bpct
    for i in range(3):
        expect(bars.nth(i).locator(".name")).to_be_visible()
        expect(bars.nth(i).locator(".bar-track .bar-fill")).to_have_count(1)
        expect(bars.nth(i).locator(".bpct")).to_be_visible()

    # confidence number present and non-empty (ends with %)
    conf = app_page.locator("#conf-body .conf-num")
    expect(conf).to_be_visible()
    assert conf.inner_text().strip().endswith("%")

    # populated status card with a valid status name
    name = app_page.locator("#tab-inference [data-status-card] .status-name")
    expect(name).to_be_visible()
    assert name.inner_text().strip() in STATUS_NAMES


def test_confidence_gauge_and_marker_render(app_page):
    expect(app_page.locator("#conf-body .gauge")).to_have_count(1)
    marker = app_page.locator("#conf-body .gauge .marker")
    expect(marker).to_have_count(1)
    # marker positioned via style.left as a percentage
    left = marker.evaluate("el => el.style.left")
    assert left.endswith("%") and left != ""


def test_conf_num_carries_status_class(app_page):
    cls = app_page.locator("#conf-body .conf-num").get_attribute("class") or ""
    tokens = cls.split()
    assert "conf-num" in tokens
    assert any(s in tokens for s in ("reliable", "uncertain", "reject")), cls


# ---------------- dashboard tab ----------------

def test_dashboard_charts_and_tables(app_page):
    app_page.click('.tab[data-tab="dashboard"]')
    app_page.wait_for_selector("#chart-reliability svg", timeout=15000)

    expect(app_page.locator("#chart-reliability svg")).to_have_count(1)
    expect(app_page.locator("#chart-risk svg")).to_have_count(1)

    # metrics table with 4 rows: ECE / NLL / Brier / Accuracy
    expect(app_page.locator("#metrics-table table.metrics")).to_have_count(1)
    rows = app_page.locator("#metrics-table table.metrics tbody tr")
    expect(rows).to_have_count(4)
    names = rows.locator(".metric-name").evaluate_all(
        "els => els.map(e => e.childNodes[0].textContent.trim())"
    )
    assert names == ["ECE", "NLL", "Brier", "Accuracy"], names

    # exactly 10 per-class rows
    expect(app_page.locator("#classacc-rows .ca-row")).to_have_count(10)


def test_dashboard_risk_svg_has_paths_and_circles(app_page):
    app_page.click('.tab[data-tab="dashboard"]')
    app_page.wait_for_selector("#chart-risk svg", timeout=15000)
    assert app_page.locator("#chart-risk svg path").count() >= 1
    assert app_page.locator("#chart-risk svg circle").count() >= 1


# ---------------- corruption lab tab ----------------

def test_corruption_tab_widgets(app_page):
    app_page.click('.tab[data-tab="corruption"]')
    expect(app_page.locator("#tab-corruption")).to_have_class("tab-panel active")

    # three sliders in the corruption group
    expect(app_page.locator('#tab-corruption input[type=range][data-group="cor"]')).to_have_count(3)

    # 4 quality meters
    expect(app_page.locator("#qmeters .qmeter")).to_have_count(4)

    # compare panel: clean baseline + corruption cell
    expect(app_page.locator("#conf-compare .cc-cell")).to_have_count(2)
    expect(app_page.locator("#conf-compare .cc-cell.clean .big")).to_be_visible()

    # live canvases
    expect(app_page.locator("#wave-cor")).to_have_count(1)
    expect(app_page.locator("#mel-cor")).to_have_count(1)


def test_corruption_reset_button_present(app_page):
    expect(app_page.locator("#tab-corruption [data-reset]")).to_have_count(1)


# ---------------- global: no uncaught pageerror ----------------

def test_no_pageerror_on_load(app_page, page_errors):
    assert _no_pageerror(page_errors) == [], page_errors


def test_no_pageerror_after_full_walkthrough(app_page, page_errors):
    for data_sample, _ in CHIPS:
        with app_page.expect_response(lambda r: "/api/predict" in r.url and r.request.method == "POST"):
            app_page.click(f'.chip[data-sample="{data_sample}"]')
    app_page.click('.tab[data-tab="corruption"]')
    app_page.click('.tab[data-tab="dashboard"]')
    app_page.wait_for_selector("#chart-reliability svg", timeout=15000)
    app_page.click('.tab[data-tab="inference"]')
    assert _no_pageerror(page_errors) == [], page_errors
