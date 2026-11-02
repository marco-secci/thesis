# Blood donations in Trieste and Gorizia, 2009–2025

**English** · [Italiano](README.it.md)

Data, code and text of my bachelor's thesis: seventeen years of blood
donations in the provinces of Trieste and Gorizia (Friuli Venezia Giulia,
Italy), from the raw extracts to a forecast
of donations for 2026–2031. One command rebuilds every table and figure in the
thesis.

> **Thesis:** *Donazioni di sangue nelle province di Trieste e Gorizia dal 2009 al 2025: analisi dei dati e studio sul futuro*
> Marco Secci, BSc in Artificial Intelligence and Data Analytics, University of Trieste, A.Y. 2025/2026
> Supervisor: Leonardo Egidi · Co-supervisor: Massimo La Raja

> **Code documentation:** [marco-secci.github.io/thesis](https://marco-secci.github.io/thesis/)
> (every module and function of the pipeline, generated from the source with Sphinx)

## Contents

- [What this repository contains](#what-this-repository-contains)
- [Acknowledgements and data source](#acknowledgements-and-data-source)
- [Reproducing the analysis](#reproducing-the-analysis)
- [The Shiny dashboard (v0.1.0, demo)](#the-shiny-dashboard-v010-demo)
- [The thesis (in Italian)](#the-thesis-in-italian)
- [Licence](#licence)
- [Citation and contact](#citation-and-contact)

## What this repository contains

- **A reproducible pipeline** (`src/`, Python): cleaning, a donor × year panel,
  descriptive analysis, retention, an inferential GLM, a rolling-origin
  comparison of 17 forecasting models, and the 2026–2031 forecast.
- **The cleaned dataset** (`data/processed/donations.parquet`): 353,671
  donations by 43,299 pseudonymous donors. The pipeline starts from it.
- **An interactive dashboard** (`shiny_app/`, R Shiny), still a demo.
- **The thesis** (`thesis/`), as LaTeX sources and a compiled PDF.

Headline results. The selected model is a damped trend (φ = 0.6), chosen on
rolling-origin backtests at 1–3 year horizons:

| model | MAPE | bias |
|---|---|---|
| **damped linear trend, φ = 0.6** | **3.55%** | −0.8% |
| quasi-Poisson monthly GLM, linear trend | 3.91% | −2.5% |
| undamped linear trend | 4.92% | −4.1% |
| naive (last observed year) | 5.17% | +3.7% |
| per-donor Poisson (simulated and summed) | 6.46% | +3.1% |
| quasi-Poisson monthly GLM, 4-df spline | 7.07% | −1.0% |

The table reports the values printed in the thesis (Appendix A). Only the
per-donor model is a Monte Carlo simulation. The thesis run used a
simulation seed that changed from one execution to the next; the pipeline now
fixes it, and gives 6.48% / +3.0% for that row. The difference is 0.02
points and leaves the ranking unchanged. Every other number reproduces
exactly.

| year | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 |
|---|---|---|---|---|---|---|
| forecast donations | 16,607 | 16,370 | 16,205 | 16,089 | 16,008 | 15,951 |

## Acknowledgements and data source

This work would not exist without **ASUGI** (Azienda Sanitaria Universitaria
Giuliano Isontina) and its **Dipartimento Interaziendale di Medicina
Trasfusionale (DIMT)** of Trieste and Gorizia. They provided the donation archive and allowed the
cleaned dataset to be published. My sincere thanks to them.

The source data are 27 Excel extracts from the DIMT information system, one or
more per year from 2009 to 2025, with 370,552 rows in total. Each row is a
donation by a pseudonymous donor. `src/ingest.py` and `src/clean.py` turn
them into the published dataset. In that dataset the date of birth is reduced
to the year of birth: everything else is as used in the thesis.

| what | where | in the repository |
|---|---|---|
| cleaned donations, 2009–2025 | `data/processed/donations.parquet` | yes, see [`data/README.md`](data/README.md) for columns and terms of use |
| resident population of Trieste and Gorizia, NUTS-3 (Eurostat `demo_r_pjanaggr3`) | `data/raw/istat/eurostat_nuts3_fvg_pjanaggr3.tsv` | yes (open data, CC BY 4.0) |
| resident population of FVG by 5-year age group, NUTS-2 (Eurostat `demo_r_pjangroup`), used only for the age profile | `data/raw/istat/eurostat_fvg_pop.json` | yes (open data, CC BY 4.0) |
| SHA-256 of the original XLSX files | `data/manifest.sha256` | yes |
| original XLSX extracts | `data/raw/XLSX/` | no, they stay with ASUGI |

Using the donation data for anything beyond reproducing this work requires
permission from the author and from ASUGI (see [Licence](#licence)).

## Reproducing the analysis

### Requirements

Only [**uv**](https://docs.astral.sh/uv/). It installs the right Python (3.12)
and the exact package versions locked in `uv.lock`, inside a local `.venv/`.
Nothing is installed system-wide.

```powershell
winget install --id=astral-sh.uv -e                 # Windows
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
```

### Run everything

From the repository root:

```powershell
.\run_pipeline.ps1                 # Windows (PowerShell)
```

```bash
./run_pipeline.sh                  # Linux, macOS, Git Bash, WSL
```

If PowerShell refuses to run scripts, use
`powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1`.

The script sets up the environment, then runs each step and shows its progress:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Blood donations in Trieste and Gorizia, 2009-2025: full pipeline
  data:  cleaned dataset data/processed/donations.parquet (ingest and clean skipped)
  steps: 12   logs: reports/logs/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ 1/12] ▶ panel           build the dense donor x year panel
        ✔ 6.2s      -> data/processed/donor_year.parquet
[ 2/12] ▶ eda             descriptive statistics and exploratory figures
        ✔ 9.8s      -> reports/
...
```

A full run takes about 10–15 minutes on a laptop. Most of that time goes to the
model comparison (`models`). The steps are:

| # | step | what it does | writes |
|---|---|---|---|
| 1 | `ingest` | merges the 27 XLSX files *(only if they are present)* | `data/interim/events_raw.parquet` |
| 2 | `clean` | applies the named cleaning rules *(only if the XLSX are present)* | `data/processed/donations.parquet` |
| 3 | `panel` | dense donor × year grid with explicit zeros | `data/processed/donor_year.parquet` |
| 4 | `eda` | descriptive statistics | `reports/` |
| 5 | `retention` | retention and cohort survival (ch. 3) | `reports/retention_*.csv` |
| 6 | `glm` | inferential GLM, IRR table (ch. 4) | `reports/glm_*.csv` |
| 7 | `dash_overdisp` | Poisson vs negative binomial at dashboard grain (app. B) | `reports/` |
| 8 | `baselines` | rolling-origin backtest of the baselines | `reports/backtest_annuale*.csv` |
| 9 | `models` | full comparison of the 17 candidate models (app. A) | `reports/backtest_modelli*.csv` |
| 10 | `forecast` | 2026–2031 forecast with empirical intervals (app. A) | `reports/forecast_2026_2031.csv` |
| 11 | `population` | donation rates per resident (Eurostat, ch. 2) | `reports/population_*.csv` |
| 12 | `figures` | every figure of the thesis | `thesis/img/` |
| 13 | `export_legacy` | the files read by the dashboard | `data/FINAL/` |
| 14 | `tests` | `pytest` invariants on the produced data | |

Without the XLSX files (the normal case), steps 1–2 are skipped and the run
starts from the published dataset. If you do have the XLSX files, put them in
`data/raw/XLSX/`. The script then checks them against `data/manifest.sha256`
and runs the whole chain.

Options are passed through to the script:

```bash
./run_pipeline.sh --list                 # list the steps
./run_pipeline.sh --from forecast        # resume from a step
./run_pipeline.sh --only figures tests   # run selected steps
./run_pipeline.sh --dashboard            # only what the dashboard needs (a few minutes)
./run_pipeline.sh --sims 100 --no-tests  # more simulations in the backtest, skip pytest
```

Each step writes its full output to `reports/logs/<step>.log`. If a step fails,
the script prints the end of that log and tells you how to resume.

### Alternatives

- **Single modules.** Every step is a plain module:
  `uv run python -m forecast`, `uv run pytest`, and so on.
- **Containers.** The `Makefile` runs the same steps inside podman (rootless)
  or Docker: `make build`, then `make all` (`make ENGINE=docker all` for
  Docker). `make help` lists every target.
- **API documentation.** It is rebuilt and published automatically on every
  push to `main`. To build it locally: `uv sync --extra dev --extra docs`, then
  `uv run sphinx-build -b html sphinx sphinx/_build/html` and open
  `sphinx/_build/html/index.html`.

## The Shiny dashboard (v0.1.0, demo)

> [!NOTE]
> **This is a demo, version 0.1.0.** It is an interactive companion to
> Appendix B of the thesis, meant for exploring the data locally. It is not a
> validated tool for operational planning. Its interface, and the numbers it
> shows, may change in later versions.

What it shows, in eight tabs:

1. **Overview**: key figures and donations per year
2. **Calendar heatmap**: donations day by day within a chosen month
3. **Age**: age distributions, young-donor retention, donations per age group
4. **Time series**: yearly, monthly and weekday patterns
5. **Univariate**: sex, donation type, collection point, donor class
6. **Data table**: filter, sort and download the rows
7. **Regression**: Poisson and negative binomial GLMs, refitted on the current filters
8. **Forecast 2026–31**: the pipeline's forecast (read from file, never recomputed)

Four global filters in the sidebar (year, sex, donation type, age range) apply to
every tab except young-donor retention, which always uses the full history.

### Install and run

1. **Install R** (4.4 or newer) from [CRAN](https://cran.r-project.org/), then
   install the packages once:

   ```r
   install.packages(c("shiny", "shinyWidgets", "dplyr", "tidyr", "lubridate",
                      "ggplot2", "scales", "DT", "MASS", "readr"))
   ```

2. **Generate the files the dashboard reads** (`data/FINAL/`) from the
   repository root. Either run the full pipeline, or just the part the
   dashboard needs:

   ```powershell
   .\run_pipeline.ps1 --dashboard
   ```

3. **Start the app** from the repository root:

   ```bash
   Rscript -e "shiny::runApp('shiny_app', port = 8181, launch.browser = TRUE)"
   ```

   It opens at <http://localhost:8181>. The first load takes a few seconds,
   because it reads the whole dataset. From RStudio you can also open
   `shiny_app/app.R` and press **Run App**.

With podman or Docker instead of a local R: `make build-r`, then `make shiny`.

## The thesis (in Italian)

The thesis is written **in Italian**. The compiled PDF is
[`thesis/Tesi.pdf`](thesis/Tesi.pdf). To rebuild it from the sources, you need
any LaTeX distribution (TeX Live, MiKTeX):

```bash
cd thesis && latexmk -pdf Tesi.tex
```

Or with containers: `make build-tex`, then `make thesis`.

**Need it in English?** If a partial or full translation would help you, for
example a chapter, the methodology, or the results, don't hesitate to ask.
[Open an issue](../../issues) or write to
**marco.secci@studenti.units.it**.

## Licence

Different parts of the repository are covered by different terms:

| part | terms |
|---|---|
| code: `src/`, `scripts/`, `tests/`, `shiny_app/`, `run_pipeline.*`, `Makefile`, `Dockerfile` | [MIT](LICENSE) |
| thesis: everything in `thesis/` (text, figures, PDF) | **all rights reserved**, see [`thesis/LICENSE`](thesis/LICENSE). You may read it and quote it with citation. You may not redistribute or modify it without permission |
| donation data: `data/processed/donations.parquet` | **all rights reserved** (ASUGI and the author), see [`data/README.md`](data/README.md#terms-of-use). Use beyond reproducing this work requires written permission from both |
| Eurostat population data | CC BY 4.0, © European Union |

## Citation and contact

```bibtex
@thesis{secci2026donazioni,
  author = {Secci, Marco},
  title  = {Donazioni di sangue nelle province di Trieste e Gorizia dal 2009 al 2025: analisi dei dati e studio sul futuro},
  type   = {Bachelor's thesis},
  school = {Universit{\`a} degli Studi di Trieste},
  year   = {2026}
}
```

Questions, bugs, data requests or translation requests: [open an
issue](../../issues) or write to **marco.secci@studenti.units.it**.
