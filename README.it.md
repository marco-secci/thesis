# Donazioni di sangue a Trieste e Gorizia, 2009–2025

[English](README.md) · **Italiano**

Dati, codice e testo della mia tesi triennale: diciassette anni di donazioni
di sangue nelle province di Trieste e Gorizia, dagli estratti grezzi alla previsione delle
donazioni per il 2026–2031. Un solo comando rigenera tutte le tabelle e le
figure della tesi.

> **Tesi:** *Donazioni di sangue nelle province di Trieste e Gorizia dal 2009 al 2025: analisi dei dati e studio sul futuro*
> Marco Secci, Laurea Triennale in Intelligenza Artificiale e Data Analytics, Università degli Studi di Trieste, A.A. 2025/2026
> Relatore: Leonardo Egidi · Correlatore: Massimo La Raja

> **Documentazione del codice (in inglese):** [marco-secci.github.io/thesis](https://marco-secci.github.io/thesis/)
> (tutti i moduli e le funzioni della pipeline, generata dal codice con Sphinx)

## Indice

- [Cosa contiene il repository](#cosa-contiene-il-repository)
- [Ringraziamenti e origine dei dati](#ringraziamenti-e-origine-dei-dati)
- [Riprodurre l'analisi](#riprodurre-lanalisi)
- [La dashboard Shiny (v0.1.0, demo)](#la-dashboard-shiny-v010-demo)
- [La tesi (in italiano)](#la-tesi-in-italiano)
- [Licenza](#licenza)
- [Citazione e contatti](#citazione-e-contatti)

## Cosa contiene il repository

- **Una pipeline riproducibile** (`src/`, Python): pulizia, panel donatore ×
  anno, analisi descrittiva, fidelizzazione, GLM inferenziale, confronto a
  origine mobile di 17 modelli di previsione e previsione 2026–2031.
- **Il dataset pulito** (`data/processed/donations.parquet`): 353.671 donazioni
  di 43.299 donatori pseudonimizzati. La pipeline riparte da qui.
- **Una dashboard interattiva** (`shiny_app/`, R Shiny), ancora una demo.
- **La tesi** (`thesis/`), in sorgenti LaTeX e PDF compilato.

Risultati principali. Il modello scelto è un trend smorzato (φ = 0,6),
selezionato con backtest a origine mobile a orizzonte 1–3 anni:

| modello | MAPE | bias |
|---|---|---|
| **trend lineare smorzato, φ = 0,6** | **3,55%** | −0,8% |
| GLM quasi-Poisson mensile, trend lineare | 3,91% | −2,5% |
| trend lineare non smorzato | 4,92% | −4,1% |
| naive (ultimo anno osservato) | 5,17% | +3,7% |
| Poisson per donatore (simulato e sommato) | 6,46% | +3,1% |
| GLM quasi-Poisson mensile, spline 4 gdl | 7,07% | −1,0% |

La tabella riporta i valori stampati nella tesi (Appendice A). Solo il
modello per donatore è una simulazione Monte Carlo. L'esecuzione della tesi
usava un seed di simulazione che cambiava da un'esecuzione all'altra; la
pipeline ora lo fissa e per quella riga dà 6,48% / +3,0%. La differenza è di
0,02 punti e non cambia la classifica. Tutti gli altri numeri si riproducono
esattamente.

| anno | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 |
|---|---|---|---|---|---|---|
| donazioni previste | 16.607 | 16.370 | 16.205 | 16.089 | 16.008 | 15.951 |

## Ringraziamenti e origine dei dati

Questo lavoro non esisterebbe senza **ASUGI** (Azienda Sanitaria Universitaria
Giuliano Isontina) e il suo **Dipartimento Interaziendale di Medicina
Trasfusionale (DIMT)** di Trieste e Gorizia. ASUGI ha fornito l'archivio delle donazioni e ha
autorizzato la pubblicazione del dataset pulito. A loro va il mio sincero
ringraziamento.

I dati di partenza sono 27 estratti Excel del sistema informativo del DIMT,
uno o più per anno dal 2009 al 2025, per un totale di 370.552 righe. Ogni riga
è una donazione di un donatore pseudonimizzato. `src/ingest.py` e
`src/clean.py` li trasformano nel dataset pubblicato. In quel dataset la data
di nascita è ridotta all'anno di nascita: tutto il resto è come nella tesi.

| cosa | dove | nel repository |
|---|---|---|
| donazioni pulite, 2009–2025 | `data/processed/donations.parquet` | sì: colonne e condizioni d'uso in [`data/README.md`](data/README.md) |
| popolazione residente di Trieste e Gorizia, NUTS-3 (Eurostat `demo_r_pjanaggr3`) | `data/raw/istat/eurostat_nuts3_fvg_pjanaggr3.tsv` | sì (dati aperti, CC BY 4.0) |
| popolazione del FVG per fascia quinquennale, NUTS-2 (Eurostat `demo_r_pjangroup`), solo per il profilo per età | `data/raw/istat/eurostat_fvg_pop.json` | sì (dati aperti, CC BY 4.0) |
| SHA-256 degli XLSX originali | `data/manifest.sha256` | sì |
| estratti XLSX originali | `data/raw/XLSX/` | no, restano ad ASUGI |

Usare i dati delle donazioni per qualcosa di diverso dalla riproduzione di
questo lavoro richiede l'autorizzazione dell'autore e di ASUGI (vedi
[Licenza](#licenza)).

## Riprodurre l'analisi

### Requisiti

Solo [**uv**](https://docs.astral.sh/uv/). Installa il Python giusto (3.12) e
le versioni esatte dei pacchetti bloccate in `uv.lock`, dentro una `.venv/`
locale. Non installa nulla a livello di sistema.

```powershell
winget install --id=astral-sh.uv -e                 # Windows
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
```

### Eseguire tutto

Dalla radice del repository:

```powershell
.\run_pipeline.ps1                 # Windows (PowerShell)
```

```bash
./run_pipeline.sh                  # Linux, macOS, Git Bash, WSL
```

Se PowerShell non permette di eseguire script, usa
`powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1`.

Lo script prepara l'ambiente, poi esegue i passi uno alla volta e ne mostra
l'avanzamento:

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

Un'esecuzione completa richiede circa 10–15 minuti su un portatile, quasi
tutti per il confronto dei modelli (`models`). I passi sono:

| # | passo | cosa fa | scrive |
|---|---|---|---|
| 1 | `ingest` | unisce i 27 XLSX *(solo se presenti)* | `data/interim/events_raw.parquet` |
| 2 | `clean` | applica le regole di pulizia nominate *(solo se gli XLSX sono presenti)* | `data/processed/donations.parquet` |
| 3 | `panel` | griglia densa donatore × anno, con gli zeri espliciti | `data/processed/donor_year.parquet` |
| 4 | `eda` | statistiche descrittive | `reports/` |
| 5 | `retention` | fidelizzazione e sopravvivenza per coorte (cap. 3) | `reports/retention_*.csv` |
| 6 | `glm` | GLM inferenziale, tabella IRR (cap. 4) | `reports/glm_*.csv` |
| 7 | `dash_overdisp` | Poisson contro binomiale negativa alla grana della dashboard (app. B) | `reports/` |
| 8 | `baselines` | backtest a origine mobile delle baseline | `reports/backtest_annuale*.csv` |
| 9 | `models` | confronto completo dei 17 modelli candidati (app. A) | `reports/backtest_modelli*.csv` |
| 10 | `forecast` | previsione 2026–2031 con intervalli empirici (app. A) | `reports/forecast_2026_2031.csv` |
| 11 | `population` | tassi di donazione per residente (Eurostat, cap. 2) | `reports/population_*.csv` |
| 12 | `figures` | tutte le figure della tesi | `thesis/img/` |
| 13 | `export_legacy` | i file letti dalla dashboard | `data/FINAL/` |
| 14 | `tests` | test `pytest` sugli invarianti dei dati prodotti | |

Senza gli XLSX (il caso normale) i passi 1–2 vengono saltati e la pipeline
parte dal dataset pubblicato. Chi ha gli XLSX li mette in `data/raw/XLSX/`: lo
script li confronta con `data/manifest.sha256` ed esegue l'intera catena.

Le opzioni vengono passate allo script:

```bash
./run_pipeline.sh --list                 # elenca i passi
./run_pipeline.sh --from forecast        # riparte da un passo
./run_pipeline.sh --only figures tests   # esegue solo alcuni passi
./run_pipeline.sh --dashboard            # solo ciò che serve alla dashboard (pochi minuti)
./run_pipeline.sh --sims 100 --no-tests  # più simulazioni nel backtest, senza pytest
```

Ogni passo scrive il proprio output completo in `reports/logs/<passo>.log`. Se
un passo fallisce, lo script mostra la fine del log e indica come ripartire.

### Alternative

- **Singoli moduli.** Ogni passo è un modulo Python:
  `uv run python -m forecast`, `uv run pytest`, e così via.
- **Container.** Il `Makefile` esegue gli stessi passi dentro podman
  (rootless) o Docker: `make build`, poi `make all` (`make ENGINE=docker all`
  per Docker). `make help` elenca tutti i target.
- **Documentazione del codice.** Viene rigenerata e pubblicata automaticamente
  a ogni push su `main`. Per generarla in locale:
  `uv sync --extra dev --extra docs`, poi
  `uv run sphinx-build -b html sphinx sphinx/_build/html` e aprire
  `sphinx/_build/html/index.html`.

## La dashboard Shiny (v0.1.0, demo)

> [!NOTE]
> **È una demo, versione 0.1.0.** Accompagna l'appendice B della tesi e serve
> a esplorare i dati in locale. Non è uno strumento validato per la
> pianificazione operativa. Interfaccia e numeri mostrati potranno cambiare
> nelle versioni successive.

Cosa mostra, in otto schede:

1. **Overview**: indicatori principali e donazioni per anno
2. **Calendar heatmap**: donazioni giorno per giorno in un mese scelto
3. **Age**: distribuzione per età, fidelizzazione dei giovani, donazioni per fascia d'età
4. **Time series**: andamento annuale, mensile e per giorno della settimana
5. **Univariate**: sesso, tipo di donazione, punto di raccolta, classe del donatore
6. **Data table**: righe filtrabili, ordinabili e scaricabili
7. **Regression**: GLM Poisson e binomiale negativa, ristimati sui filtri correnti
8. **Forecast 2026–31**: la previsione della pipeline (letta da file, mai ricalcolata)

I quattro filtri globali nella barra laterale (anno, sesso, tipo di donazione, fascia
d'età) valgono per tutte le schede tranne la fidelizzazione dei giovani, che
usa sempre la storia completa.

### Installazione e avvio

1. **Installa R** (4.4 o successivo) da [CRAN](https://cran.r-project.org/),
   poi installa i pacchetti, una volta sola:

   ```r
   install.packages(c("shiny", "shinyWidgets", "dplyr", "tidyr", "lubridate",
                      "ggplot2", "scales", "DT", "MASS", "readr"))
   ```

2. **Genera i file letti dalla dashboard** (`data/FINAL/`) dalla radice del
   repository. Puoi eseguire la pipeline completa, oppure solo la parte che
   serve alla dashboard:

   ```powershell
   .\run_pipeline.ps1 --dashboard
   ```

3. **Avvia l'app** dalla radice del repository:

   ```bash
   Rscript -e "shiny::runApp('shiny_app', port = 8181, launch.browser = TRUE)"
   ```

   Si apre su <http://localhost:8181>. Il primo caricamento richiede qualche
   secondo, perché legge l'intero dataset. Da RStudio puoi anche aprire
   `shiny_app/app.R` e premere **Run App**.

Con podman o Docker al posto di un R locale: `make build-r`, poi `make shiny`.

## La tesi (in italiano)

La tesi è scritta **in italiano**. Il PDF compilato è
[`thesis/Tesi.pdf`](thesis/Tesi.pdf). Per ricompilarlo dai sorgenti basta una
qualunque distribuzione LaTeX (TeX Live, MiKTeX):

```bash
cd thesis && latexmk -pdf Tesi.tex
```

Oppure con i container: `make build-tex`, poi `make thesis`.

**Serve in inglese?** Se ti è utile una traduzione parziale o completa, per
esempio un capitolo, la metodologia o i risultati, non esitare a chiedere:
[apri una issue](../../issues) o scrivi a **marco.secci@studenti.units.it**.

## Licenza

Le diverse parti del repository hanno condizioni diverse:

| parte | condizioni |
|---|---|
| codice: `src/`, `scripts/`, `tests/`, `shiny_app/`, `run_pipeline.*`, `Makefile`, `Dockerfile` | [MIT](LICENSE) |
| tesi: tutto il contenuto di `thesis/` (testo, figure, PDF) | **tutti i diritti riservati**, vedi [`thesis/LICENSE`](thesis/LICENSE). Si può leggere e citare indicando la fonte; non si può ridistribuire né modificare senza autorizzazione |
| dati delle donazioni: `data/processed/donations.parquet` | **tutti i diritti riservati** (ASUGI e l'autore), vedi [`data/README.md`](data/README.md#terms-of-use). Ogni uso diverso dalla riproduzione di questo lavoro richiede l'autorizzazione scritta di entrambi |
| dati di popolazione Eurostat | CC BY 4.0, © Unione Europea |

## Citazione e contatti

```bibtex
@thesis{secci2026donazioni,
  author = {Secci, Marco},
  title  = {Donazioni di sangue nelle province di Trieste e Gorizia dal 2009 al 2025: analisi dei dati e studio sul futuro},
  type   = {Tesi di laurea triennale},
  school = {Universit{\`a} degli Studi di Trieste},
  year   = {2026}
}
```

Domande, bug, richieste di dati o di traduzione: [apri una issue](../../issues)
o scrivi a **marco.secci@studenti.units.it**.
