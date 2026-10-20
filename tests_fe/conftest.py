"""Shared harness for the SoundGuard frontend (Playwright) test suites.

Each pytest process starts its OWN uvicorn on a free port (session-scoped), so several suites can run
concurrently without colliding. Fixtures:

    base_url       -> "http://127.0.0.1:<port>" of a live, healthy server serving app.py
    page           -> a fresh Playwright page (from pytest-playwright); does NOT navigate
    page_errors    -> list that accumulates ("console"|"pageerror", type, text); register-before-navigate
    app_page       -> a page already navigated to "/" with the first prediction rendered (happy-path start)

Fault-injection tests that must intercept a route BEFORE load should use `page` + `base_url` and call
page.route(...) themselves, then page.goto(base_url + "/").

The browser is launched with a fake media device and the context is granted the microphone permission,
so the Record button's getUserMedia + MediaRecorder path can be exercised headless.

DOM contract (selectors the app renders into) is documented in tests_fe/DOM_CONTRACT.md.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def live_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=PROJECT_ROOT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 45
    healthy = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=1) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            time.sleep(0.4)
    if not healthy:
        proc.terminate()
        raise RuntimeError("server did not become healthy in time")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def base_url(live_server):
    return live_server


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    # fake media stream so getUserMedia/MediaRecorder work without real hardware
    return {
        **browser_type_launch_args,
        "args": [
            *browser_type_launch_args.get("args", []),
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
        ],
    }


@pytest.fixture
def browser_context_args(browser_context_args):
    return {**browser_context_args, "permissions": ["microphone"]}


@pytest.fixture
def page_errors(page):
    """Accumulate console errors and uncaught page errors. Handlers are registered here, so request
    this fixture (directly or via app_page) BEFORE navigation to catch load-time errors."""
    errs = []
    page.on("console", lambda m: errs.append(("console", m.type, m.text)) if m.type == "error" else None)
    page.on("pageerror", lambda e: errs.append(("pageerror", "", str(e))))
    return errs


@pytest.fixture
def app_page(page, base_url, page_errors):
    """A page navigated to '/' with the initial prediction rendered — the normal starting state."""
    page.goto(base_url + "/")
    page.wait_for_selector("#pred-body .top1", timeout=20000)
    return page
