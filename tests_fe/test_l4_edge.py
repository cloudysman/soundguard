"""L4 — adversarial data, malformed 2xx bodies, timing/races and boundaries.

Every test's baseline assertion is: the frontend must NOT throw an uncaught `pageerror` that breaks
the page. The app's real error path is `refresh()`'s try/catch → `showError()`; a malformed body should
therefore degrade to a "Backend error" status card or a best-effort render, never an uncaught crash.

Selectors verified against static/index.html + soundguard-app.js. See tests_fe/DOM_CONTRACT.md.
"""
import json
import pathlib
import re
import time

import pytest
from playwright.sync_api import expect


# ---------------------------------------------------------------- fixtures/helpers
CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
SAMPLE_IDX = {'dog': 3, 'siren': 8, 'jack': 7, 'horn': 1, 'kids': 2, 'gun': 6, 'ac': 0}

_DEFAULT_ORDER = [[3, 0.9], [0, 0.04], [1, 0.02], [2, 0.01], [4, 0.008],
                  [5, 0.007], [6, 0.006], [7, 0.005], [8, 0.004], [9, 0.0]]


def _probs_from(order):
    p = [0.0] * 10
    for i, v in order:
        if isinstance(i, int) and 0 <= i < 10:
            p[i] = v
    return p


ORDER_JSON = json.dumps(_DEFAULT_ORDER)
PROBS_JSON = json.dumps(_probs_from(_DEFAULT_ORDER))


def build(conf='0.9', entropy='0.15', snr='30.0', lvl='-6.0', clip='0.0',
          status='"reliable"', cause='"clean"', order=None, probs=None,
          include_order=True, include_probs=True):
    """Assemble a prediction body from RAW JSON tokens so we can inject NaN/Infinity/strings/nulls."""
    order = ORDER_JSON if order is None else order
    probs = PROBS_JSON if probs is None else probs
    parts = []
    if include_probs:
        parts.append(f'"probs":{probs}')
    if include_order:
        parts.append(f'"order":{order}')
    parts += [f'"conf":{conf}', f'"entropy":{entropy}', f'"snrDb":{snr}',
              f'"levelDb":{lvl}', f'"clipPct":{clip}', f'"status":{status}', f'"cause":{cause}']
    return '{' + ','.join(parts) + '}'


def _order_for(idx):
    rest = [i for i in range(10) if i != idx][:9]
    ps = [0.04, 0.02, 0.01, 0.008, 0.007, 0.006, 0.005, 0.004, 0.0]
    lst = [[idx, 0.9]] + [[i, p] for i, p in zip(rest, ps)]
    return json.dumps(lst)


def _body_for(idx):
    return build(order=_order_for(idx))


def _pageerrors(errs):
    return [e for e in errs if e[0] == 'pageerror']


def _wait_until(page, cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        page.wait_for_timeout(25)
    return cond()


# =================================================================== malformed 2xx bodies
_ORDER_OOB = json.dumps([[99, 0.9], [0, 0.04], [1, 0.02]])           # index outside 0..9
_ORDER_15 = json.dumps([[i % 10, 0.9 - i * 0.02] for i in range(15)])  # > 10 classes

MALFORMED = [
    ('order_missing',       build(include_order=False)),
    ('probs_missing',       build(include_probs=False)),
    ('conf_null',           build(conf='null')),
    ('conf_nan_literal',    build(conf='NaN')),          # invalid JSON token
    ('conf_infinity_token', build(conf='Infinity')),     # invalid JSON token
    ('conf_huge_1e309',     build(conf='1e309')),         # parses to Infinity
    ('order_empty',         build(order='[]')),
    ('order_index_oob',     build(order=_ORDER_OOB)),
    ('order_more_than_10',  build(order=_ORDER_15)),
    ('strings_for_numbers', build(conf='"high"', entropy='"lots"', snr='"loud"',
                                  lvl='"x"', clip='"y"')),
]


@pytest.mark.parametrize('name,body', MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_predict_no_pageerror(page, base_url, page_errors, name, body):
    """A 2xx /api/predict with a broken body must be handled (render or Backend-error card),
    never an uncaught pageerror that kills the page."""
    page.route('**/api/predict',
               lambda route: route.fulfill(status=200, content_type='application/json', body=body))
    page.goto(base_url + '/')
    # Both the normal render and showError() write a .status-name into every [data-status-card];
    # waiting on it proves refresh() has resolved one way or the other.
    page.wait_for_selector('[data-status-card] .status-name', timeout=15000)
    assert _pageerrors(page_errors) == [], (name, page_errors)


def test_malformed_then_recovers(page, base_url, page_errors):
    """After a malformed body, un-routing and selecting a real chip must recover a genuine render."""
    page.route('**/api/predict',
               lambda route: route.fulfill(status=200, content_type='application/json',
                                           body=build(include_order=False)))
    page.goto(base_url + '/')
    page.wait_for_selector('[data-status-card] .status-name', timeout=15000)
    page.unroute('**/api/predict')
    page.click('.chip[data-sample="dog"]')
    expect(page.locator('#pred-body .top1 .label')).to_have_text('dog bark', timeout=15000)
    assert _pageerrors(page_errors) == [], page_errors


# =================================================================== the tag names the cause
CAUSE_TAGS = [
    ('uncertain', 'quiet',  '1.0',  'level too low'),
    ('uncertain', 'uncert', '0.5',  'model hesitating'),
    ('reject',    'noise',  '0.95', 'noise floor too close'),
    ('reject',    'clip',   '0.99', 'samples clipped'),
    ('reliable',  'clean',  '0.93', 'input accepted'),
]


@pytest.mark.parametrize('status,cause,conf,tag', CAUSE_TAGS,
                         ids=[f'{s}_{c}' for s, c, _, _ in CAUSE_TAGS])
def test_status_tag_names_the_cause_not_the_state(page, base_url, page_errors,
                                                  status, cause, conf, tag):
    """The short tag under the verdict must name the cause the backend reported, not the state.

    The two axes of the policy are combined by or, so the recording axis can raise the verdict
    while the model is at full confidence. A tag fixed to the state then read "high uncertainty"
    beside a confidence of 100 percent — the interface asserting something the measurement does not
    support, which is the class of defect appendix F of the report records. The state is already
    carried by the name and the colour of the card, so the tag is free to carry the cause.
    """
    body = build(conf=conf, status=f'"{status}"', cause=f'"{cause}"')
    page.route('**/api/predict',
               lambda route: route.fulfill(status=200, content_type='application/json', body=body))
    page.goto(base_url + '/')
    page.wait_for_selector('[data-status-card] .status-name', timeout=15000)
    expect(page.locator('[data-status-card] .status-tag').first).to_have_text(tag)
    assert _pageerrors(page_errors) == [], page_errors


# =================================================================== meters agree with the verdict
def test_meter_thresholds_match_the_shipped_policy():
    """The quality meters colour three numbers the rejection policy also thresholds.

    They are two copies of the same constants, one in Python and one in JavaScript, and nothing at
    run time reconciles them. When they drifted apart a meter turned red on a recording the policy
    called reliable, which is the interface contradicting the verdict printed beside it.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    js = (root / 'static' / 'soundguard-data.js').read_text(encoding='utf-8')
    py = (root / 'rejection.py').read_text(encoding='utf-8')

    block = re.search(r'SG\.gateThresholds\s*=\s*\{(.*?)\n\};', js, re.S)
    assert block, 'SG.gateThresholds is gone from soundguard-data.js'
    front = {}
    for meter, bad, warn in re.findall(
            r'(\w+):\s*\{\s*bad:\s*(-?[\d.]+),\s*warn:\s*(-?[\d.]+)', block.group(1)):
        front[f'{meter}_bad'] = float(bad)
        front[f'{meter}_warn'] = float(warn)
    assert len(front) == 6, f'expected three meters with two cut-offs each, read {front}'

    def const(name):
        m = re.search(rf'^\s*{name}:\s*float\s*=\s*(-?[\d.]+)', py, re.M)
        assert m, f'{name} is gone from rejection.py'
        return float(m.group(1))

    for meter, poor, weak in (('snr', 'snr_poor', 'snr_weak'),
                              ('lvl', 'level_poor', 'level_weak'),
                              ('clip', 'clip_poor', 'clip_weak')):
        assert front[f'{meter}_bad'] == const(poor), meter
        assert front[f'{meter}_warn'] == const(weak), meter


# =================================================================== the caption follows the picture
def test_mel_caption_says_so_when_no_spectrogram_was_returned(page, base_url, page_errors):
    """The spectrogram card is captioned "what the model sees". That is true only of the measured
    spectrogram carried by the response. When the response has none the card draws a placeholder,
    and the caption must stop making the claim — otherwise the panel asserts a measurement that was
    never taken, which is the defect appendix F of the report records.
    """
    page.route('**/api/predict',
               lambda route: route.fulfill(status=200, content_type='application/json',
                                           body=build()))          # no mel field
    page.goto(base_url + '/')
    page.wait_for_selector('[data-status-card] .status-name', timeout=15000)
    caption = page.locator('#mel-inf').locator('xpath=ancestor::div[contains(@class,"card")][1]') \
                  .locator('.card-head .hint')
    expect(caption).not_to_have_text('what the model sees')
    assert 'illustration' in (caption.text_content() or '').lower()
    assert _pageerrors(page_errors) == [], page_errors


# =================================================================== debounce coalescing
def test_slider_spam_debounces_to_last_value(app_page, page_errors):
    """~30 rapid input events on one slider must coalesce (80 ms debounce) to a single refresh
    reflecting the LAST value — one refresh == exactly two POST /api/predict (current + clean)."""
    app_page.click('.tab[data-tab="corruption"]')
    sel = 'input[data-slider="noise"][data-group="cor"]'
    app_page.wait_for_selector(sel)

    predicts = []
    app_page.on('request',
                lambda r: predicts.append(r.url)
                if (r.method == 'POST' and r.url.endswith('/api/predict')) else None)

    # 30 synchronous input events (values 2..60) then a final settle on 77.
    app_page.locator(sel).evaluate(
        "el => { for (let i=1;i<=30;i++){ el.value=String(i*2);"
        "el.dispatchEvent(new Event('input',{bubbles:true})); }"
        "el.value='77'; el.dispatchEvent(new Event('input',{bubbles:true})); }")

    # label updates synchronously to the last value regardless of debounce
    expect(app_page.locator('[data-sval="noise-cor"]')).to_have_text('77%')

    # wait for the single coalesced refresh (2 predicts) then confirm no further ones sneak in
    _wait_until(app_page, lambda: len(predicts) >= 2, timeout=3.0)
    app_page.wait_for_timeout(250)
    assert len(predicts) == 2, f'expected 1 coalesced refresh (2 predicts), got {len(predicts)}'
    assert _pageerrors(page_errors) == [], page_errors


# =================================================================== boundary slider values
def test_boundary_slider_0_and_100(app_page, page_errors):
    """Slider extremes 0 and 100 must produce a clean render, correct label, and no pageerror."""
    app_page.click('.tab[data-tab="corruption"]')
    sel = 'input[data-slider="noise"][data-group="cor"]'
    app_page.wait_for_selector(sel)
    for val in ('100', '0'):
        with app_page.expect_response(
                lambda r: r.request.method == 'POST' and r.url.endswith('/api/predict')):
            app_page.locator(sel).evaluate(
                f"el => {{ el.value='{val}';"
                f"el.dispatchEvent(new Event('input',{{bubbles:true}})); }}")
        expect(app_page.locator('[data-sval="noise-cor"]')).to_have_text(f'{val}%')
        # pred-body lives on the (hidden) inference tab; assert the corruption-tab render instead
        app_page.wait_for_selector('#conf-compare .cc-cell.clean .big')
        assert app_page.inner_text('#conf-compare .cc-cell.clean .big').strip() != ''
        expect(app_page.locator('#tab-corruption [data-status-card] .status-name')).not_to_have_text('')
    assert _pageerrors(page_errors) == [], page_errors


# =================================================================== timing / races
def _install_stash(page, base_url):
    """Route /api/predict: pass initial load through to the real backend, then STASH later requests
    so the test can fulfil them in an arbitrary order. Returns (stash, flip) where flip() turns on
    stashing and stash is a list of (sample, route)."""
    ctl = {'stash': False}
    stash = []

    def handler(route):
        if not ctl['stash']:
            route.continue_()
            return
        try:
            sample = (route.request.post_data_json or {}).get('sample')
        except Exception:
            sample = None
        stash.append((sample, route))

    page.route('**/api/predict', handler)
    page.goto(base_url + '/')
    page.wait_for_selector('#pred-body .top1', timeout=20000)

    def flip():
        ctl['stash'] = True
    return stash, flip


def _run_out_of_order(page, base_url):
    """Select siren then gun fast, then deliver replies out of order (gun first, siren last).
    Returns the settled top-1 label text."""
    stash, flip = _install_stash(page, base_url)
    flip()

    page.click('.chip[data-sample="siren"]')   # first selection (2 predicts stashed)
    page.click('.chip[data-sample="gun"]')     # last selection  (2 predicts stashed)

    assert _wait_until(page, lambda: len(stash) >= 4, timeout=5.0), \
        f'expected 4 stashed predicts, got {len(stash)}'

    # deliver the LAST-selected (gun) FIRST, then the FIRST-selected (siren) LAST → out of order
    for s, r in stash:
        if s == 'gun':
            r.fulfill(status=200, content_type='application/json', body=_body_for(SAMPLE_IDX['gun']))
    for s, r in stash:
        if s == 'siren':
            r.fulfill(status=200, content_type='application/json', body=_body_for(SAMPLE_IDX['siren']))

    # wait for the DOM to stop changing (both refreshes applied) so the read is deterministic
    page.wait_for_function(
        "() => { const l=document.querySelector('#pred-body .top1 .label');"
        "return l && ['siren','gun shot'].includes(l.textContent); }", timeout=6000)
    page.wait_for_timeout(200)
    return page.inner_text('#pred-body .top1 .label').strip()


def test_out_of_order_no_pageerror(page, base_url, page_errors):
    """Out-of-order chip replies must never raise an uncaught pageerror; the UI still settles to a
    real label."""
    label = _run_out_of_order(page, base_url)
    assert label in ('siren', 'gun shot'), label
    assert _pageerrors(page_errors) == [], page_errors


def test_out_of_order_last_selected_chip_wins(page, base_url, page_errors):
    """Correct behaviour: the LAST-selected chip (gun → 'gun shot') should be what renders."""
    label = _run_out_of_order(page, base_url)
    assert label == 'gun shot', f'stale-render race: last selection was gun but UI shows {label!r}'


def test_very_slow_response_still_resolves(page, base_url, page_errors):
    """A ~2 s delayed /api/predict must still resolve and render — no timeout, no crash."""
    stash, flip = _install_stash(page, base_url)
    flip()

    page.click('.chip[data-sample="horn"]')
    assert _wait_until(page, lambda: len(stash) >= 2, timeout=5.0), \
        f'expected 2 stashed predicts, got {len(stash)}'

    page.wait_for_timeout(2000)  # the slow network
    for _s, r in stash:
        r.fulfill(status=200, content_type='application/json', body=_body_for(SAMPLE_IDX['horn']))

    expect(page.locator('#pred-body .top1 .label')).to_have_text('car horn', timeout=6000)
    assert _pageerrors(page_errors) == [], page_errors


# =================================================================== record button
def test_record_start_then_stop_no_error(app_page, page_errors):
    """Start recording, then stop — label returns to 'Record', no uncaught pageerror.
    (The webm upload is rejected by the backend as a normal Backend-error card, which is fine.)"""
    btn = app_page.locator('#btn-record')
    btn.click()
    app_page.wait_for_selector('#btn-record.recording', timeout=6000)
    btn.click()
    app_page.wait_for_function(
        "() => document.querySelector('#btn-record .lbl').textContent==='Record'", timeout=6000)
    assert _pageerrors(page_errors) == [], page_errors


def test_record_rapid_double_click_no_error(app_page, page_errors):
    """Two very fast clicks on the record button must not raise an uncaught pageerror."""
    btn = app_page.locator('#btn-record')
    btn.click()
    btn.click()
    app_page.wait_for_timeout(400)
    assert _pageerrors(page_errors) == [], page_errors


def test_record_label_says_how_to_stop(app_page, page_errors):
    """While recording, the label must name the action the next click performs.

    The same button starts and stops, so a label that only counts elapsed time leaves a user with
    no way to discover how to end the recording. A real user hit exactly that: the timer ran past
    forty seconds because nothing on the button said it was also the stop control. Asserting the
    text keeps the affordance from being refactored away.
    """
    btn = app_page.locator('#btn-record')
    label = app_page.locator('#btn-record .lbl')
    assert label.text_content() == 'Record'

    btn.click()
    app_page.wait_for_selector('#btn-record.recording', timeout=6000)
    app_page.wait_for_timeout(300)

    during = label.text_content()
    assert 'Stop' in during, f"the label must offer to stop; it read {during!r}"
    assert btn.get_attribute('aria-label') == 'Stop recording'

    btn.click()
    app_page.wait_for_function(
        "() => document.querySelector('#btn-record .lbl').textContent==='Record'", timeout=8000)
    assert btn.get_attribute('aria-label') == 'Record from the microphone'
    assert _pageerrors(page_errors) == [], page_errors


def test_record_stops_itself_at_the_cap(app_page, page_errors):
    """Recording ends on its own, without a second click.

    The preprocessing keeps only the first few seconds, so a recording left running is discarded
    almost entirely. The cap bounds it whether or not the user finds the stop control.
    """
    app_page.locator('#btn-record').click()
    app_page.wait_for_selector('#btn-record.recording', timeout=6000)
    app_page.wait_for_function(
        "() => document.querySelector('#btn-record .lbl').textContent==='Record'", timeout=20000)
    assert app_page.locator('#btn-record.recording').count() == 0
    assert _pageerrors(page_errors) == [], page_errors
