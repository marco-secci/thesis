# ================= #
# RUN_PIPELINE FILE #
# ================= #
"""Run the whole pipeline with one command, showing progress step by step.

Same steps and order as `make all`, plus the Shiny export, but without make,
podman or Docker: every step is `python -m <module>` with `src/` on the path.
Each step's output goes to `reports/logs/<step>.log`; on failure the tail of
that log is printed and the run stops.

Normally launched through the wrappers, which also install the environment:

    ./run_pipeline.sh  [options]        # Linux, macOS, Git Bash, WSL
    .\\run_pipeline.ps1 [options]        # Windows PowerShell

Options::

    --list            show the steps and exit
    --from STEP       start from STEP (earlier outputs must already exist)
    --only STEP ...   run only the given steps
    --dashboard       only what the Shiny dashboard needs (a few minutes)
    --sims N          simulations per backtest origin (default 30)
    --no-tests        skip pytest at the end

Classes and functions
---------------------
All of them are used only inside this script; no other module imports it.
It is launched by ``run_pipeline.sh`` and ``run_pipeline.ps1``.

_c
    Builds the terminal colouring helpers (``bold``, ``red``, ...).
Step
    Dataclass describing one step; :meth:`Step.command` gives its command
    line.
steps
    The ordered list of steps.
fmt_time
    Formats a duration for display.
run_step
    Runs one step, logging its output.
banner
    Prints the header of the run.
summary
    Prints the final status table.
main
    Command-line entry point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOGS = ROOT / "reports" / "logs"
sys.path.insert(0, str(SRC))

import config  # noqa: E402  (needs SRC on the path)


# ── Terminal styling ─────────────────────────────────────────────────────────

if os.name == "nt":
    os.system("")  # turns on ANSI escape handling in the Windows console

COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


# ========= #
# _C METHOD #
# ========= #
def _c(code: str):
    """Build an ANSI colouring function for one SGR code.

    Parameters
    ----------
    code : str
        ANSI SGR code, e.g. ``"1"`` (bold) or ``"31"`` (red).

    Returns
    -------
    callable
        Function ``str -> str`` that wraps its argument in the escape codes,
        or returns it unchanged when ``COLOR`` is false.
    """
    return (lambda s: f"\033[{code}m{s}\033[0m") if COLOR else (lambda s: s)


bold, dim, red, green, yellow, cyan = (_c(c) for c in ("1", "2", "31", "32", "33", "36"))

try:
    "▶✔✘·━".encode(sys.stdout.encoding or "utf-8")
    ARROW, OK, KO, SKIP, RULE = "▶", "✔", "✘", "·", "━"
except UnicodeEncodeError:
    ARROW, OK, KO, SKIP, RULE = ">", "OK", "X", "-", "="


# ── Steps ────────────────────────────────────────────────────────────────────

# ========== #
# STEP CLASS #
# ========== #
@dataclass
class Step:
    """One pipeline step, run as ``python -m <module>``.

    Attributes
    ----------
    name : str
        Step name used on the command line (``--from``, ``--only``) and for
        the log file.
    module : str
        Module under ``src/`` to run, or ``"pytest"`` for the test suite.
    what : str
        One-line description printed while running.
    writes : str
        Main output path, printed on success.
    needs_raw : bool, default False
        Whether the step needs the raw XLSX files (skipped without them).
    args : list of str
        Extra command-line arguments for the module.
    """
    name: str
    module: str
    what: str
    writes: str
    needs_raw: bool = False
    args: list[str] = field(default_factory=list)

    # ============== #
    # COMMAND METHOD #
    # ============== #
    def command(self) -> list[str]:
        """Command line that runs this step with the current interpreter.

        Returns
        -------
        list of str
            ``[python, "-m", "pytest"]`` for the test step, otherwise
            ``[python, "-m", module, *args]``.
        """
        if self.module == "pytest":
            return [sys.executable, "-m", "pytest"]
        return [sys.executable, "-m", self.module, *self.args]


# ============ #
# STEPS METHOD #
# ============ #
def steps(sims: int) -> list[Step]:
    """The full, ordered list of pipeline steps.

    Parameters
    ----------
    sims : int
        Simulations per backtest origin, passed to ``backtest_models`` as
        ``--sims``.

    Returns
    -------
    list of Step
        From ``ingest`` to ``tests``, in execution order.
    """
    return [
        Step("ingest", "ingest", "merge the 27 raw XLSX files into one event table",
             "data/interim/events_raw.parquet", needs_raw=True),
        Step("clean", "clean", "apply the named cleaning rules",
             "data/processed/donations.parquet", needs_raw=True),
        Step("panel", "panel", "build the dense donor x year panel",
             "data/processed/donor_year.parquet"),
        Step("eda", "eda", "descriptive statistics and exploratory figures", "reports/"),
        Step("retention", "retention", "donor retention and cohort survival (ch. 3)",
             "reports/retention_*.csv"),
        Step("glm", "glm_inference", "inferential GLM at donor-year level (ch. 4)",
             "reports/glm_*.csv"),
        Step("dash_overdisp", "dash_overdisp", "Poisson vs NB check at dashboard grain (app. B)",
             "reports/"),
        Step("baselines", "baselines", "rolling-origin backtest of the baselines",
             "reports/backtest_annuale*.csv"),
        Step("models", "backtest_models", f"full model comparison ({sims} sims per origin)",
             "reports/backtest_modelli*.csv", args=["--sims", str(sims)]),
        Step("forecast", "forecast", "2026-2031 forecast with empirical intervals (app. A)",
             "reports/forecast_2026_2031.csv"),
        Step("population", "population", "donation rates per resident (Eurostat, ch. 2)",
             "reports/population_*.csv"),
        Step("figures", "figures_tesi", "regenerate every thesis figure", "thesis/img/"),
        Step("export_legacy", "export_legacy", "CSV export read by the Shiny dashboard",
             "data/FINAL/"),
        Step("tests", "pytest", "invariant tests on the produced data", "-"),
    ]


#: Minimum chain for `data/FINAL/`: the forecast needs the baseline backtest
#: (for its empirical intervals), the export needs the forecast.
DASHBOARD_STEPS = ["panel", "baselines", "forecast", "export_legacy"]


# ── Running ──────────────────────────────────────────────────────────────────

# =============== #
# FMT_TIME METHOD #
# =============== #
def fmt_time(s: float) -> str:
    """Format a duration for display.

    Parameters
    ----------
    s : float
        Duration in seconds.

    Returns
    -------
    str
        ``"12.3s"`` below one minute, otherwise ``"4m 05s"``.
    """
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60):02d}s"


# =============== #
# RUN_STEP METHOD #
# =============== #
def run_step(i: int, n: int, step: Step) -> tuple[bool, float]:
    """Run one step, showing a live timer and saving its output to a log.

    The step's stdout and stderr go to ``reports/logs/<step>.log``; on
    failure the last 30 lines of the log are printed.

    Parameters
    ----------
    i : int
        1-based position of the step in the plan.
    n : int
        Number of steps in the plan.
    step : Step
        Step to run.

    Returns
    -------
    ok : bool
        True if the step exited with code 0.
    elapsed : float
        Run time in seconds.
    """
    head = f"{dim(f'[{i:>2}/{n}]')} {cyan(ARROW)} {bold(step.name):<{24 if COLOR else 15}} {dim(step.what)}"
    print(head)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{step.name}.log"
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8",
           "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}

    t0 = time.time()
    done = threading.Event()

    def ticker():
        while not done.wait(0.5):
            if COLOR:
                print(f"\r        {yellow('running')} {dim(fmt_time(time.time() - t0))}  ",
                      end="", flush=True)

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(step.command(), cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    done.set()
    th.join()
    dt = time.time() - t0
    if COLOR:
        print("\r\033[K", end="")

    if rc == 0:
        print(f"        {green(OK)} {fmt_time(dt):<9} {dim('-> ' + step.writes)}")
        return True, dt

    print(f"        {red(KO + ' failed')} after {fmt_time(dt)} (exit code {rc})")
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
    print(red("        " + RULE * 3 + f" last lines of {log.relative_to(ROOT)} " + RULE * 3))
    for line in tail:
        print(red(f"        {line}"))
    return False, dt


# ============= #
# BANNER METHOD #
# ============= #
def banner(mode: str, plan: list[Step]) -> None:
    """Print the header with the data source and the number of steps.

    Parameters
    ----------
    mode : str
        Description of the data the run starts from.
    plan : list of Step
        Steps that will run.

    Returns
    -------
    None
    """
    line = RULE * 64
    print()
    print(cyan(line))
    print(bold("  Blood donations in Trieste and Gorizia, 2009-2025: full pipeline"))
    print(f"  {dim('data:')}  {mode}")
    print(f"  {dim('steps:')} {len(plan)}   {dim('logs:')} {LOGS.relative_to(ROOT)}{os.sep}")
    print(cyan(line))
    print()


# ============== #
# SUMMARY METHOD #
# ============== #
def summary(results: list[tuple[Step, str, float]], total: float) -> None:
    """Print the final table of steps with status and time.

    Parameters
    ----------
    results : list of tuple (Step, str, float)
        Step, status (``"ok"``, ``"failed"`` or ``"skipped"``) and run time
        in seconds.
    total : float
        Total run time in seconds.

    Returns
    -------
    None
    """
    print()
    print(cyan(RULE * 64))
    for step, status, dt in results:
        mark = {"ok": green(OK), "failed": red(KO), "skipped": dim(SKIP)}[status]
        t = fmt_time(dt) if status != "skipped" else ""
        print(f"  {mark} {step.name:<15} {t:>9}")
    print(cyan(RULE * 64))
    print(f"  total time: {bold(fmt_time(total))}")


# =========== #
# MAIN METHOD #
# =========== #
def main() -> int:
    """Command-line entry point: choose the steps and run them in order.

    Decides where the pipeline can start from (raw XLSX files, or the
    cleaned dataset with ingest and clean skipped), checks the raw files
    against the manifest on a full run, then runs the planned steps,
    stopping at the first failure. Options are described in the module
    docstring.

    Returns
    -------
    int
        Process exit code: 0 on success (or ``--list``), 1 if no data is
        found or a step fails, 2 for an unknown step name.
    """
    ap = argparse.ArgumentParser(description="Run the whole data pipeline.")
    ap.add_argument("--list", action="store_true", help="show the steps and exit")
    ap.add_argument("--from", dest="start", metavar="STEP", help="start from STEP")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    ap.add_argument("--dashboard", action="store_true",
                    help="only the steps the Shiny dashboard needs")
    ap.add_argument("--sims", type=int, default=30, help="simulations per backtest origin")
    ap.add_argument("--no-tests", action="store_true", help="skip pytest at the end")
    args = ap.parse_args()

    all_steps = steps(args.sims)
    names = [s.name for s in all_steps]

    if args.list:
        for i, s in enumerate(all_steps, 1):
            raw = yellow(" (needs raw XLSX)") if s.needs_raw else ""
            print(f"  {i:>2}. {bold(s.name):<{24 if COLOR else 15}} {s.what}{raw}")
        return 0

    for n in [args.start] if args.start else (args.only or []):
        if n not in names:
            print(red(f"unknown step '{n}'. Available: {', '.join(names)}"))
            return 2

    # Which data is available decides where the pipeline can start from.
    has_raw = bool(config.raw_files())
    has_clean = config.DONATIONS.exists()
    if has_raw:
        mode = f"raw XLSX files in {config.RAW_XLSX.relative_to(ROOT)}"
    elif has_clean:
        mode = f"cleaned dataset {config.DONATIONS.relative_to(ROOT)} (ingest and clean skipped)"
    else:
        print(red(f"\n  No data found.\n"))
        print(f"  The pipeline needs either the raw XLSX files in {config.RAW_XLSX.relative_to(ROOT)}"
              f"\n  or the cleaned dataset {config.DONATIONS.relative_to(ROOT)}.")
        print("  The donation data is not public: see 'Data availability' in README.md.\n")
        return 1

    plan = all_steps
    if args.dashboard:
        args.only = DASHBOARD_STEPS
    if args.only:
        plan = [s for s in all_steps if s.name in args.only]
    elif args.start:
        plan = all_steps[names.index(args.start):]
    if args.no_tests:
        plan = [s for s in plan if s.name != "tests"]
    skipped = []
    if not has_raw:
        skipped = [s for s in plan if s.needs_raw]
        plan = [s for s in plan if not s.needs_raw]

    banner(mode, plan)

    if has_raw and not args.only and not args.start:
        import manifest
        print(f"{dim('[ 0]')} {cyan(ARROW)} {bold('manifest'):<{24 if COLOR else 15}} "
              f"{dim('check raw files against data/manifest.sha256')}")
        problems = manifest.verify()
        if problems:
            for p in problems:
                print(yellow(f"        ! {p}"))
            print(yellow("        the results will not match the thesis exactly; continuing"))
        else:
            print(f"        {green(OK)} all {len(config.raw_files())} files match")

    t_all = time.time()
    results = [(s, "skipped", 0.0) for s in skipped]
    for i, step in enumerate(plan, 1):
        ok, dt = run_step(i, len(plan), step)
        results.append((step, "ok" if ok else "failed", dt))
        if not ok:
            results += [(s, "skipped", 0.0) for s in plan[i:]]
            summary(results, time.time() - t_all)
            print(red(f"\n  Stopped at '{step.name}'. Fix the error, then resume with:"))
            print(f"    --from {step.name}\n")
            return 1

    summary(results, time.time() - t_all)
    print(f"\n  {green('Done.')} Outputs: reports/, thesis/img/, data/FINAL/ (for the dashboard).")
    shiny = """Rscript -e "shiny::runApp('shiny_app', port = 8181)\""""
    print(f"  Start the dashboard with:  {bold(shiny)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
