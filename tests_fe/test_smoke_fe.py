"""Harness smoke test — proves the live-server + Playwright wiring works before the real suites run."""


def test_page_loads_and_predicts(app_page, page_errors):
    assert "SoundGuard" in app_page.title()
    # the initial prediction rendered
    label = app_page.inner_text("#pred-body .top1 .label")
    assert label.strip() != ""
    # default chip is dog, and dog classifies as dog_bark
    assert label.strip() == "dog bark"
    # a status card is present
    assert app_page.locator("[data-status-card] .status-name").first.inner_text().strip() != ""
    # no uncaught page errors during load
    assert [e for e in page_errors if e[0] == "pageerror"] == [], page_errors


def test_dashboard_charts_render(app_page):
    app_page.click('.tab[data-tab="dashboard"]')
    app_page.wait_for_selector("#chart-reliability svg", timeout=10000)
    assert app_page.locator("#chart-risk svg").count() == 1
    assert app_page.locator("#metrics-table table").count() == 1
    assert app_page.locator("#classacc-rows .ca-row").count() == 10
