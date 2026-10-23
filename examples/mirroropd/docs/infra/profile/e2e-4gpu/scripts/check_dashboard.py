# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline dashboard browser checks. Run from repository root; no GPU experiments."""

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright
from render import render

repo = Path.cwd()
docs = repo / "examples/mirroropd/docs/infra/profile/e2e-4gpu"

spec = importlib.util.spec_from_file_location("fixtures", repo / "tests/utils/test_e2e_report.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
out = docs / "browser-check"
out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as temporary, sync_playwright() as pw:
    root = Path(temporary)
    job = fixture.write_fixture(root)
    data = {
        "generated_at": "SYNTHETIC TEST ONLY",
        "run_status": {"state": "completed"},
        "jobs": {"baseline": fixture.collect.collect_job(job)},
    }
    render(data, root / "synthetic.html")
    browser = pw.chromium.launch(headless=True, args=["--disable-gpu"])
    context = browser.new_context(viewport={"width": 1440, "height": 1100}, offline=True)
    page = context.new_page()
    errors = []
    network = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: network.append(r.url) if r.url.startswith("http") else None)
    shutil.copyfile(docs / "dashboard.html", root / "report.html")
    for file in [root / "report.html", root / "synthetic.html"]:
        page.goto(file.as_uri())
        page.wait_for_timeout(1200)
        for tab in ["overview", "rollout", "training", "evidence"]:
            page.locator("button[data-tab=" + tab + "]").click()
            page.wait_for_timeout(200)
        if file.name == "synthetic.html":
            page.locator("#step").select_option("1")
            page.locator("button[data-tab=rollout]").click()
            page.wait_for_timeout(600)
            assert "100" in page.locator("#request-table").inner_text()
            page.locator("button[data-tab=overview]").click()
            page.wait_for_timeout(500)
            page.screenshot(path=str(out / "synthetic-test.png"), full_page=True)
        else:
            assert "快照时间" in page.locator("#state").inner_text()
            page.locator("button[data-tab=overview]").click()
            page.wait_for_timeout(300)
            page.screenshot(path=str(out / "report-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(300)
            assert page.evaluate("document.documentElement.scrollWidth") <= 390
            page.screenshot(path=str(out / "report-mobile.png"), full_page=True)
            page.set_viewport_size({"width": 1440, "height": 1100})
    browser.close()
    assert not errors, errors
    assert not network, network
    print(
        json.dumps(
            {"page_errors": errors, "external_requests": network, "desktop_mobile_and_synthetic_filters": "passed"}
        )
    )
