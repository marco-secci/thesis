# ======================== #
# CAPTURE_SCREENSHOTS FILE #
# ======================== #
"""Capture the dashboard screenshots used as figures in Appendix B of the thesis.

Produces thesis/img/fig_shiny_*.png. Needs the dashboard already serving and
`playwright` installed (it drives the Edge that ships with Windows, so no
`playwright install` step is required):

    Rscript -e "shiny::runApp('shiny_app', port=8181)"
    python shiny_app/capture_screenshots.py

Optional argument: the dashboard URL (default http://localhost:8181).

The five figures are captured with the Year filter on "All", so the KPI cards
show the whole-dataset totals quoted in Appendix B. Re-running overwrites them
in place.

Functions
---------
Both are used only inside this script; no other module imports it, and it
is run by hand.

settle
    Waits until Shiny has finished recomputing.
main
    Captures and saves the screenshots.
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8181"
OUT = Path(__file__).resolve().parents[1] / "thesis" / "img"

VIEWPORT_W = 1600
MAX_H = 4000


# ============= #
# SETTLE METHOD #
# ============= #
def settle(page, timeout=180):
    """Block until Shiny has no recalculation pending.

    Shiny sets .shiny-busy on <html> while any output is being recomputed;
    without this the screenshot catches half-drawn plots.

    Parameters
    ----------
    page : playwright.sync_api.Page
        Page showing the dashboard.
    timeout : float, default 180
        Maximum seconds to wait for the busy flag to clear; after that the
        function returns anyway.

    Returns
    -------
    None
    """
    t0 = time.time()
    time.sleep(1.5)
    while time.time() - t0 < timeout:
        if not page.evaluate("document.documentElement.classList.contains('shiny-busy')"):
            break
        time.sleep(0.4)
    time.sleep(1.5)


# =========== #
# MAIN METHOD #
# =========== #
def main():
    """Open the dashboard in headless Edge and save the five screenshots.

    Sets the Year filter to "All", then visits the Overview, Calendar,
    Young Donor Retention, Regression and Forecast tabs and saves each one,
    cropped to its content, as ``thesis/img/fig_shiny_*.png``. The nested
    helpers ``tabs``, ``go`` and ``shot`` list the tabs, switch tab and wait,
    and take the cropped screenshot.

    Returns
    -------
    None
    """
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(
            viewport={"width": VIEWPORT_W, "height": 1200}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        settle(page)

        def tabs():
            return page.query_selector_all(".nav-tabs > li > a")

        def go(idx):
            tabs()[idx].click()
            settle(page)

        def shot(name):
            # Crop to the real content box so no figure carries dead space.
            height = page.evaluate(
                "Math.ceil(Math.max(...Array.from(document.querySelectorAll('body > *'))"
                ".map(e => e.getBoundingClientRect().bottom + window.scrollY))) + 16")
            height = max(600, min(height, MAX_H))
            page.set_viewport_size({"width": VIEWPORT_W, "height": height})
            time.sleep(0.8)
            page.screenshot(
                path=str(OUT / name),
                clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": height})
            page.set_viewport_size({"width": VIEWPORT_W, "height": 1200})
            print("wrote", OUT / name)

        page.evaluate("document.getElementById('year_filter').selectize.setValue('All', false)")
        settle(page)

        go(0)
        shot("fig_shiny_overview.png")
        go(1)
        shot("fig_shiny_calendar.png")
        go(2)                       # Età
        tabs()[9].click()           # -> Young Donor Retention sub-tab
        settle(page)
        shot("fig_shiny_retention.png")
        go(6)
        shot("fig_shiny_regression.png")
        go(7)
        shot("fig_shiny_forecast.png")

        browser.close()


if __name__ == "__main__":
    main()
