"""L5 — responsive / layout across viewports.

For a spread of mobile→desktop widths (320, 375, 414, 768, 1024, 1440) the app is loaded fresh at
that viewport and the layout is checked for:

  * no horizontal overflow of the document itself,
  * the header brand + tab-nav being visible,
  * every sample chip being present and fully inside the viewport (chips are expected to wrap, not clip),
  * the inference prediction card sitting inside the viewport,
  * the waveform / mel canvases having a non-zero rendered size,
  * the dashboard reliability/risk SVGs and the metrics table not overflowing their own container.

Previously-documented defect, now FIXED:
  The sticky header (`.app-header`) used to be a non-wrapping flexbox, so the brand block plus the
  three `white-space:nowrap` tab buttons (pushed right by `margin-left:auto` on `nav.tabs`) forced
  an intrinsic min-width of ~719px and the whole document overflowed horizontally below that width
  (399px @320, 344px @375, 305px @414). The fix set `.app-header`/`.tabs` to `flex-wrap: wrap`, so
  the header now wraps and there is no document horizontal overflow at any tested width. The
  `test_no_document_horizontal_overflow` params for 320/375/414 are therefore normal passing tests.
"""
import pytest
from playwright.sync_api import expect

WIDTHS = [320, 375, 414, 768, 1024, 1440]
HEIGHT = 900

# Header now wraps (flex-wrap: wrap) — no document horizontal overflow at any width.
_OVERFLOW_PARAMS = WIDTHS


def _load_inference(page, base_url, width):
    """Fresh load at `width`, inference tab, first prediction rendered."""
    page.set_viewport_size({"width": width, "height": HEIGHT})
    page.goto(base_url + "/")
    page.wait_for_selector("#pred-body .top1", timeout=20000)


def _no_pageerrors(page_errors):
    return [e for e in page_errors if e[0] == "pageerror"]


# --------------------------------------------------------------------------------------
# 1. Document-level horizontal overflow
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", _OVERFLOW_PARAMS)
def test_no_document_horizontal_overflow(page, base_url, page_errors, width):
    _load_inference(page, base_url, width)
    dims = page.evaluate(
        "() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})"
    )
    assert _no_pageerrors(page_errors) == [], page_errors
    overflow = dims["sw"] - dims["cw"]
    assert dims["sw"] <= dims["cw"] + 1, (
        f"document overflows horizontally at width={width}: "
        f"scrollWidth={dims['sw']} > clientWidth={dims['cw']} (overflow {overflow}px)"
    )


# --------------------------------------------------------------------------------------
# 2. Header brand + tab nav visible at every width
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", WIDTHS)
def test_header_brand_and_tabs_visible(page, base_url, width):
    _load_inference(page, base_url, width)
    expect(page.locator(".brand .brand-name")).to_be_visible()
    expect(page.locator("nav.tabs")).to_be_visible()
    # all three tab buttons present and visible
    tabs = page.locator("nav.tabs .tab")
    assert tabs.count() == 3
    for i in range(3):
        expect(tabs.nth(i)).to_be_visible()
    assert page.locator(".brand .brand-name").inner_text().strip() == "SoundGuard"


# --------------------------------------------------------------------------------------
# 3. Sample chips: all present and fully within the viewport (they wrap, they don't clip)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", WIDTHS)
def test_sample_chips_present_and_within_viewport(page, base_url, width):
    _load_inference(page, base_url, width)
    chips = page.locator(".chip[data-sample]")
    assert chips.count() == 7, "all seven sample chips must render"
    vw = page.viewport_size["width"]
    for i in range(chips.count()):
        chip = chips.nth(i)
        expect(chip).to_be_visible()
        bb = chip.bounding_box()
        assert bb is not None, f"chip {i} has no box at width={width}"
        assert bb["width"] > 0 and bb["height"] > 0, f"chip {i} collapsed at width={width}: {bb}"
        assert bb["x"] >= -1, f"chip {i} clipped off the left at width={width}: x={bb['x']}"
        right = bb["x"] + bb["width"]
        assert right <= vw + 1, (
            f"chip {i} clipped past the right edge at width={width}: right={right:.1f} > viewport={vw}"
        )


# --------------------------------------------------------------------------------------
# 4. Inference prediction card + canvases inside the viewport with non-zero size
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", WIDTHS)
def test_prediction_card_within_viewport(page, base_url, width):
    _load_inference(page, base_url, width)
    vw = page.viewport_size["width"]
    card = page.locator("#pred-body").locator(
        "xpath=ancestor::div[contains(concat(' ', normalize-space(@class), ' '), ' card ')][1]"
    )
    bb = card.bounding_box()
    assert bb is not None, f"prediction card has no box at width={width}"
    assert bb["width"] > 0, f"prediction card collapsed at width={width}"
    assert bb["x"] >= -1, f"prediction card clipped left at width={width}: x={bb['x']}"
    right = bb["x"] + bb["width"]
    assert right <= vw + 1, (
        f"prediction card overflows the viewport at width={width}: right={right:.1f} > viewport={vw}"
    )


@pytest.mark.parametrize("width", WIDTHS)
def test_inference_canvases_have_nonzero_size(page, base_url, width):
    _load_inference(page, base_url, width)
    for sel in ("#wave-inf", "#mel-inf"):
        canvas = page.locator(sel)
        expect(canvas).to_be_visible()
        bb = canvas.bounding_box()
        assert bb is not None, f"{sel} has no box at width={width}"
        assert bb["width"] > 0 and bb["height"] > 0, (
            f"{sel} rendered with zero size at width={width}: {bb}"
        )
        # backing store must also be non-zero (intrinsic attributes)
        wh = page.eval_on_selector(sel, "c => [c.width, c.height]")
        assert wh[0] > 0 and wh[1] > 0, f"{sel} backing store is zero at width={width}: {wh}"


# --------------------------------------------------------------------------------------
# 5. Dashboard: SVG charts + metrics table must not overflow their own container
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", WIDTHS)
def test_dashboard_charts_no_container_overflow(page, base_url, page_errors, width):
    _load_inference(page, base_url, width)
    page.click('.tab[data-tab="dashboard"]')
    page.wait_for_selector("#chart-reliability svg", timeout=10000)
    page.wait_for_selector("#metrics-table table", timeout=10000)
    page.wait_for_selector("#classacc-rows .ca-row", timeout=10000)

    assert _no_pageerrors(page_errors) == [], page_errors

    measures = page.evaluate(
        """() => {
            const ids = ['chart-reliability','chart-risk','metrics-table','classacc-rows'];
            const out = {};
            for (const id of ids){
                const el = document.getElementById(id);
                out[id] = {sw: el.scrollWidth, cw: el.clientWidth};
            }
            const svg = document.querySelector('#chart-reliability svg');
            const sb = svg.getBoundingClientRect();
            out._relSvg = {w: Math.round(sb.width), h: Math.round(sb.height)};
            const t = document.querySelector('#metrics-table table');
            out._table = {sw: t.scrollWidth, ow: t.offsetWidth};
            return out;
        }"""
    )

    # each chart/table container must not have hidden horizontal overflow
    for id_ in ("chart-reliability", "chart-risk", "metrics-table", "classacc-rows"):
        m = measures[id_]
        assert m["sw"] <= m["cw"] + 1, (
            f"#{id_} overflows its container horizontally at width={width}: "
            f"scrollWidth={m['sw']} > clientWidth={m['cw']}"
        )
    # the table must fit inside its wrapper (no un-scrollable table overflow)
    assert measures["_table"]["sw"] <= measures["metrics-table"]["cw"] + 1, (
        f"metrics <table> wider than wrapper at width={width}: "
        f"table.scrollWidth={measures['_table']['sw']} > wrapper.clientWidth={measures['metrics-table']['cw']}"
    )
    # the reliability SVG must have a real rendered size
    assert measures["_relSvg"]["w"] > 0 and measures["_relSvg"]["h"] > 0, (
        f"reliability SVG rendered with zero size at width={width}: {measures['_relSvg']}"
    )
