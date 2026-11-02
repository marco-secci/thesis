# =============================================================================
# Blood Donations — Shiny Dashboard (Thesis Project)
# =============================================================================
# Purpose:
#   Interactive exploration of the cleaned blood-donation dataset
#   (data/FINAL/dataframe_cleaned.csv, 353,671 records, 2009-2025).
#   Eight tabs total:
#
#     1. Overview          -> KPIs + donations per year
#     2. Calendar Heatmap  -> per-day counts within a chosen month
#     3. Age (Età)         -> full age & retention analysis (3 sub-tabs):
#          3a. Age Overview       — distributions, gender, type, cohort
#          3b. Young Donor Retention — mirrors src/retention.py
#          3c. Donations per Age Group — frequency & volume by age
#     4. Time Series       -> yearly / monthly / weekday trends
#     5. Univariate        -> gender, type, collection point, class
#     6. Data Table        -> sortable / filterable / downloadable rows
#     7. Regression        -> Poisson & NB GLMs, refitted on the fly
#     8. Forecast 2026-31  -> the exported forecast, read from
#                             data/FINAL/forecasts/ (never recomputed here)
#
# Python modules this app mirrors / extends (src/):
#   - eda.py           — descriptive statistics behind the Age Overview sub-tab
#   - retention.py     — young-donor cohort retention (Cap. 3 of the thesis)
#   - glm_inference.py — Poisson / Negative Binomial regression (Cap. 4)
#   - forecast.py      — the 2026-2031 projection exported to data/FINAL/forecasts/
#
# Design notes:
#   - Global sidebar filters drive every tab EXCEPT the Young Donor Retention
#     sub-tab (which uses the full longitudinal dataset — truncating by year
#     would falsely shorten measured retention periods).
#   - All heavy aggregation happens inside reactive()s that are cached across
#     outputs; no computation is repeated on every UI paint.
#   - Calendar heatmap uses Date >= start & Date <= end filtering (not just
#     month(Date)==month) to prevent cross-year leakage.
# =============================================================================


# ---- Packages ---------------------------------------------------------------
library(shiny)
library(shinyWidgets)   # pickerInput with search + multi-select
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(scales)         # axis formatters: comma(), percent(), …
library(DT)             # interactive data tables
library(MASS)           # glm.nb() for Negative Binomial
library(readr)          # read_csv — ~5-10x faster than read.csv on 50 MB files


# ---- Data path resolution ---------------------------------------------------
data_path <- "../data/FINAL/dataframe_cleaned.csv"
if (!file.exists(data_path)) data_path <- "data/FINAL/dataframe_cleaned.csv"


# ---- Shared constants -------------------------------------------------------
# Italian month names — displayed in the calendar title regardless of locale.
MONTH_LABELS_IT  <- c("Gennaio","Febbraio","Marzo","Aprile","Maggio","Giugno",
                       "Luglio","Agosto","Settembre","Ottobre","Novembre","Dicembre")

# Weekday labels hard-coded Monday-first so Windows-Italian locale never
# outputs "lun/mar/..." instead of "Mon/Tue/...".
WEEKDAY_LABELS_EN <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")

# Shared ggplot theme — applied to every plot for visual consistency.
theme_thesis <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_text(face = "bold", hjust = 0.5),
      plot.subtitle    = element_text(hjust = 0.5, color = "grey35"),
      legend.title     = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
}

# Helper: small info box displayed below each plot with plain-text commentary.
# 'txt' can be a character string or a list of tagList-compatible elements.
commentary_box <- function(...) {
  div(class = "well well-sm",
      style = "margin-top:6px; font-size:0.93em; color:#333;",
      ...)
}

# Forecast outputs produced by src/forecast.py + export_legacy.py
forecast_dir <- "../data/FINAL/forecasts"
if(!dir.exists(forecast_dir)) {
  forecast_dir <- "data/FINAL/forecasts"
}
fc_paths <- list(
  daily          = file.path(forecast_dir, "daily_forecast_2026_2031.csv"),
  monthly        = file.path(forecast_dir, "monthly_forecast_2026_2031.csv"),
  type           = file.path(forecast_dir, "donation_type_forecast_2026_2031.csv"),
  newd           = file.path(forecast_dir, "new_donors_forecast_2026_2031.csv"),
  ret            = file.path(forecast_dir, "retention_forecast_2026_2031.csv"),
  donors_monthly = file.path(forecast_dir, "monthly_donors_forecast_2026_2031.csv"),
  summary        = file.path(forecast_dir, "yearly_summary_2026_2031.csv"),
  monthly_hist   = file.path(forecast_dir, "monthly_history_corrected.csv")
)
forecast_available <- all(sapply(
  fc_paths[c("daily","monthly","type","newd","ret","donors_monthly","summary")],
  file.exists))

ui <- fluidPage(

  # Custom CSS -----------------------------------------------------------------
  # .kpi-box    : overview & retention KPI cards
  # .alert-info : styled commentary headers at the top of each section
  tags$head(tags$style(HTML("
    .kpi-box { background:#f8f9fa; border-left:5px solid #2c3e50;
               padding:12px 16px; border-radius:6px; margin-bottom:10px; }
    .kpi-box .kpi-label { color:#6c757d; font-size:0.88em; }
    .kpi-box .kpi-value { color:#2c3e50; font-size:1.55em; font-weight:700; }
    .section-header { background:#d9eaf7; border-left:5px solid #2980b9;
                      padding:10px 14px; border-radius:5px; margin-bottom:14px; }
    .section-header h4 { margin:0 0 4px 0; }
  "))),

  titlePanel("Blood Donations — Thesis Dashboard (v0.1.0 demo)",
             windowTitle = "Blood Donations — Thesis Dashboard"),
  uiOutput("data_missing"),

  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Global Filters"),
      helpText("All tabs react to these filters (except the Young Donor",
               " Retention sub-tab, which needs the full longitudinal range)."),

      # Year — choices populated dynamically once data is loaded.
      selectInput("year_filter",   "Year:",   choices = NULL),

      # Gender.
      selectInput("gender_filter", "Gender:",
                  choices  = c("All","M","F","Non specificato"),
                  selected = "All"),

      # Donation type — multi-select with search.
      pickerInput("type_filter", "Donation Type(s):",
                  choices = NULL, multiple = TRUE,
                  options = list(`actions-box`  = TRUE,
                                 `live-search`  = TRUE)),

      # Age range — clamped to the real data min/max on load.
      sliderInput("age_range", "Age range:",
                  min = 18, max = 70, value = c(18, 70), step = 1),
      hr(),
      helpText("Tip: the Calendar tab shows day-level activity; Time Series",
               " shows broad yearly/seasonal patterns.")
    ),

    mainPanel(
      width = 9,
      tabsetPanel(

        # ===================================================================
        # Tab 1 - Overview
        # KPI cards + donations per year. The yearly bar deliberately ignores
        # the Year filter and highlights the selected year instead, so the
        # single-year view never loses its context.
        # ===================================================================
        tabPanel("Overview",
          br(),
          div(class="section-header",
            h4("Donation activity at a glance"),
            p("The four cards summarise the rows currently selected by the",
              " sidebar filters. The bar chart below shows the whole",
              " 2009-2025 series with every filter applied", em("except"),
              " the year, so a single-year selection is always readable",
              " against the full history (the selected year is highlighted).",
              " With no filters applied the cards reproduce the headline",
              " figures of Chapter 2: 353,671 donation records from",
              " 43,299 distinct donors, mean age 42.4 years.")
          ),
          fluidRow(
            column(3, div(class="kpi-box",
                          div(class="kpi-label","Donations"),
                          div(class="kpi-value",
                              textOutput("kpi_donations", inline=TRUE)))),
            column(3, div(class="kpi-box",
                          div(class="kpi-label","Distinct donors"),
                          div(class="kpi-value",
                              textOutput("kpi_donors", inline=TRUE)))),
            column(3, div(class="kpi-box",
                          div(class="kpi-label","Mean age"),
                          div(class="kpi-value",
                              textOutput("kpi_avg_age", inline=TRUE)))),
            column(3, div(class="kpi-box",
                          div(class="kpi-label","Volume (L)"),
                          div(class="kpi-value",
                              textOutput("kpi_volume", inline=TRUE))))
          ),
          br(),
          plotOutput("overview_yearly", height="360px"),
          commentary_box(
            strong("Donations per year."),
            " Whole-blood collection dominates the series throughout.",
            " Donors are identified by the", code("crd"), " column rather than",
            " the alternative", code("id_anag"), " (missing in the 2021-2023",
            " files), and the two overlapping 2023 extracts are merged on the",
            " event key — both decisions made by",
            code("src/ingest.py"), "; see Chapter 1 of the thesis."
          )
        ),


        # ===================================================================
        # Tab 2 - Calendar Heatmap
        # ===================================================================
        tabPanel("Calendar Heatmap", 
                 br(),
                 p("Displays daily donation volumes for the selected month. Redder days denote higher donation counts, while bluer days denote lower counts."),
                 div(style = "display:flex; justify-content:center; align-items:center; gap:20px; margin-top: 10px; margin-bottom: 20px;",
                     actionButton("prev_year",  "", icon = icon("angles-left"),  class = "btn-default"),
                     actionButton("prev_month", "", icon = icon("chevron-left"), class = "btn-primary"),
                     h3(textOutput("month_year_text", inline = TRUE), style="margin:0; width:200px; text-align:center;"),
                     actionButton("next_month", "", icon = icon("chevron-right"), class = "btn-primary"),
                     actionButton("next_year",  "", icon = icon("angles-right"), class = "btn-default")
                 ),
                 plotOutput("calendar_plot", height = "500px")),

        # ===================================================================
        # Tab 3 - Age (Età)
        # Three sub-tabs: distributions, young-donor retention, frequency.
        # ===================================================================
        tabPanel("Età",
          br(),
          tabsetPanel(

            # ---------------------------------------------------------------
            # Sub-tab 3a — Age Overview
            # Mirrors the age & gender distributions from src/eda.py
            # ---------------------------------------------------------------
            tabPanel("Age Overview",
              br(),
              div(class="section-header",
                h4("Age distribution of donors"),
                p("Italian blood donation regulations (D.Lgs 261/2007 + AVIS guidelines)",
                  " set a minimum age of 18 and a maximum of 70 years.",
                  " Across the 2009-2025 dataset the mean age at donation is",
                  strong("42.4 years"), " (median 44), reflecting an established rather than youthful donor base.",
                  " Understanding age structure is fundamental for designing targeted recruitment",
                  " campaigns and modelling future supply under demographic change.",
                  " All four plots below react to the global sidebar filters.")
              ),
              fluidRow(
                column(6,
                  plotOutput("age_histogram", height="330px"),
                  commentary_box(
                    strong("Age distribution of donations in scope."),
                    " Each bar covers a 2-year band. The vertical dashed line marks",
                    " the weighted mean. The distribution is roughly unimodal and",
                    " right-skewed, with the bulk of donations concentrated between",
                    " 30 and 55 years. Very few donors appear below 20 or above 65,",
                    " consistent with the legal age bounds and natural drop-off in",
                    " late-career donors."
                  )
                ),
                column(6,
                  plotOutput("age_by_gender", height="330px"),
                  commentary_box(
                    strong("Age split by gender (boxplot)."),
                    " Across the full dataset, male donors (M) outnumber female donors (F)",
                    " by roughly 1.6:1 in unique-donor counts (26,474 M vs 16,512 F,",
                    " Chapter 2 of the thesis). The median ages are similar,",
                    " but the male distribution tends to extend further into the",
                    " 50-65 age range, whereas females show a slightly earlier peak."
                  )
                )
              ),
              br(),
              fluidRow(
                column(6,
                  plotOutput("age_by_type", height="330px"),
                  commentary_box(
                    strong("Age by donation type (boxplot)."),
                    " Different modalities attract distinct age profiles.",
                    " Whole-blood (SANGUE) donations cover a broad age span,",
                    " while plasmapheresis (PLASMAF) tends to skew younger —",
                    " plasma donation protocols (more frequent, shorter sessions)",
                    " appeal to younger donors with more flexible schedules.",
                    " Apheresis types (CITOAFEN, PIASTR) may show narrower ranges",
                    " depending on eligibility criteria."
                  )
                ),
                column(6,
                  plotOutput("age_by_cohort", height="330px"),
                  commentary_box(
                    strong("Mean age by birth decade over years."),
                    " Each line tracks one birth cohort (decade) as it ages through",
                    " the 2009-2025 observation window. Lines slope upward at ~1 yr/yr",
                    " if the cohort keeps donating (natural ageing). A cohort that",
                    " disappears early from the plot has dropped out of the donor pool.",
                    " Cohorts with fewer than 20 observations in any year are excluded",
                    " to suppress noise."
                  )
                )
              )
            ), # end sub-tab 3a

            # ---------------------------------------------------------------
            # Sub-tab 3b — Young Donor Retention
            # Mirrors the cohort retention analysis in src/retention.py
            # ---------------------------------------------------------------
            tabPanel("Young Donor Retention",
              br(),
              div(class="section-header",
                h4("Do young donors keep donating?"),
                p("This sub-tab mirrors the analysis in",
                  em("src/retention.py"), " (Chapter 3 of the thesis).",
                  " We identify donors whose", strong("first observed donation"),
                  " occurred when they were between 18 and 25 years old (adjustable",
                  " via the slider below), then measure their",
                  strong("retention period"), " — the number of calendar years",
                  " between their first and last observed donation in the dataset",
                  " (0 = only one calendar year of activity; 16 = active across",
                  " the full 2009-2025 window)."),
                p(strong("Important:"), " because this analysis tracks each donor",
                  " longitudinally across years, the", em("Year filter"),
                  " in the sidebar is intentionally", strong("not applied here"),
                  " — restricting to a single year would artificially truncate all",
                  " retention measurements. The Gender filter IS applied so you can",
                  " compare male vs female retention patterns.")
              ),

              # Controls + KPIs in one row
              fluidRow(
                column(4,
                  wellPanel(
                    sliderInput("young_age_range",
                                "First-donation age range:",
                                min=18, max=30, value=c(18,25), step=1),
                    hr(),
                    helpText("Default 18-25 matches the standard 'young donor'",
                             " definition used in Italian blood service recruitment.",
                             " Extend to 18-30 to capture late-starters.")
                  )
                ),
                column(8,
                  fluidRow(
                    column(3, div(class="kpi-box",
                                  div(class="kpi-label","Young donors"),
                                  div(class="kpi-value",
                                      textOutput("kpi_young_n", inline=TRUE)))),
                    column(3, div(class="kpi-box",
                                  div(class="kpi-label","One-time only"),
                                  div(class="kpi-value",
                                      textOutput("kpi_one_time", inline=TRUE)))),
                    column(3, div(class="kpi-box",
                                  div(class="kpi-label","Retained ≥ 1 yr"),
                                  div(class="kpi-value",
                                      textOutput("kpi_retained_1", inline=TRUE)))),
                    column(3, div(class="kpi-box",
                                  div(class="kpi-label","Retained ≥ 5 yrs"),
                                  div(class="kpi-value",
                                      textOutput("kpi_retained_5", inline=TRUE))))
                  )
                )
              ),
              br(),

              # Retention bar + pie
              fluidRow(
                column(7,
                  plotOutput("retention_bar", height="360px"),
                  commentary_box(
                    strong("Retention period distribution (bar chart)."),
                    " Each bar represents the number of young donors whose first",
                    " and last recorded donations are exactly N years apart.",
                    " The sharp drop from year 0 to year 1 is a structural feature",
                    " of voluntary donation cohorts worldwide: a large fraction of",
                    " first-time donors never return: the bar at 0 is by far",
                    " the tallest, and the KPI cards above quantify it for the",
                    " age band currently selected.",
                    " The distribution has a long right tail, with some donors",
                    " active across the full 2009-2025 observation window.",
                    br(), br(),
                    strong("Why these figures differ from Chapter 3."),
                    " This tab derives the cohort from the ages actually",
                    " observed in the dataset (a donor is 'young' if the",
                    " youngest age on any of their records falls in the",
                    " selected band), over the whole 2009-2025 window.",
                    " The notebook behind Chapter 3 instead cohorts on the",
                    " recorded ", code("first_donation_date"), " and stops at",
                    " the 2023 entry cohort. The two populations therefore do",
                    " not coincide, and neither do the percentages."
                  )
                ),
                column(5,
                  plotOutput("retention_pie", height="360px"),
                  commentary_box(
                    strong("One-time vs retained (pie)."),
                    " A coarser but intuitive split: donors who appear in only",
                    " one year ('one-time') versus those who return at least once",
                    " ('retained'). This mirrors the 'Option A' pie chart in the",
                    " notebook. A majority of first-time young donors do come",
                    " back at least once, but the margin is narrow — see the",
                    " caveat under the bar chart before comparing these shares",
                    " with Chapter 3."
                  )
                )
              ),
              br(),

              # Cumulative retention curve
              fluidRow(
                column(12,
                  plotOutput("retention_cumulative", height="320px"),
                  commentary_box(
                    strong("Cumulative retention curve."),
                    " The y-axis shows the", em("proportion"), " of young donors",
                    " who donated for", em("at least N years."),
                    " At N = 0 this is 100% (trivially, all donors are active for",
                    " ≥ 0 years). At N = 1 this equals the overall retention rate.",
                    " This is analogous to a Kaplan-Meier survival curve, but",
                    " simplified: we observe only the first and last years, not",
                    " exact event times. The steep initial decline reflects the",
                    " large one-time cohort; the curve then flattens among the",
                    " committed long-term donors."
                  )
                )
              ),
              br(),

              # Retention distribution table
              h4("Retention Distribution Table"),
              p("Detailed numbers behind the chart above, one row per retention",
                " period (in years)."),
              DTOutput("retention_table")
            ), # end sub-tab 3b

            # ---------------------------------------------------------------
            # Sub-tab 3c — Donations per Age Group
            # Extends the frequency-by-age analysis in src/eda.py with
            # a total-volume scatter.
            # ---------------------------------------------------------------
            tabPanel("Donations per Age Group",
              br(),
              div(class="section-header",
                h4("Donation frequency and total volume by age"),
                p("This sub-tab addresses a simple question:",
                  " do older donors give more frequently?",
                  " In the cleaned dataset, each row is", strong("one donation event"),
                  " (so the row count per donor-year = that donor's yearly frequency).",
                  " The", em("num_donations"), " column contains the cumulative total",
                  " as of the record date, which is not used here.",
                  " Both plots react to the global sidebar filters.")
              ),
              fluidRow(
                column(6,
                  plotOutput("age_freq_bar", height="380px"),
                  commentary_box(
                    strong("Average yearly donations by 5-year age band."),
                    " For each donor-year pair we count events, then average",
                    " across all donors in the same age band.",
                    " Established donors (45-60 years) often have well-practiced",
                    " annual routines, leading to slightly higher average frequency.",
                    " Young donors (18-25) show lower frequency, consistent with",
                    " the high one-time rate observed in the retention analysis.",
                    " Note: the 'log scale' used in the notebook for visual clarity",
                    " is replaced here with a linear scale for interpretability."
                  )
                ),
                column(6,
                  plotOutput("age_totalvol_scatter", height="380px"),
                  commentary_box(
                    strong("Total donated volume (L) vs. donor age — LOESS trend."),
                    " Each point represents one donor; volume is the sum of all",
                    " quantity_ml records (converted to litres) within the filtered",
                    " scope. Because older donors have had more years to accumulate",
                    " donations, total volume generally increases with age — but",
                    " variability is very high (some young donors who donate plasma",
                    " frequently can accumulate comparable volumes).",
                    " Up to 5,000 donors are sampled at random for rendering speed.",
                    " The red LOESS band gives the smoothed conditional mean ± 95% CI."
                  )
                )
              )
            ) # end sub-tab 3c
          ) # end tabsetPanel for Age
        ), # end Tab 3 — Age (Età)


        # ===================================================================
        # Tab 4 — Time Series
        # ===================================================================
        tabPanel("Time Series",
          br(),
          fluidRow(column(12, plotOutput("ts_yearly",  height="300px"))),
          br(),
          fluidRow(
            column(6, plotOutput("ts_monthly",  height="300px")),
            column(6, plotOutput("ts_weekday",  height="300px"))
          )
        ),

        # ===================================================================
        # Tab 5 — Univariate distributions
        # ===================================================================
        tabPanel("Univariate",
          br(),
          fluidRow(
            column(6, plotOutput("gender_pie",     height="320px")),
            column(6, plotOutput("type_pie",       height="320px"))
          ),
          br(),
          fluidRow(
            column(6, plotOutput("collection_bar", height="340px")),
            column(6, plotOutput("class_bar",      height="340px"))
          )
        ),

        # ===================================================================
        # Tab 6 — Data Table
        # ===================================================================
        tabPanel("Data Table",
          br(),
          fluidRow(
            column(9,
                   pickerInput("dt_cols","Columns to display:",
                               choices=NULL, multiple=TRUE,
                               options=list(`actions-box`=TRUE,
                                            `live-search`=TRUE,
                                            `selected-text-format`="count > 3"))),
            column(3, br(),
                   downloadButton("dt_download","Download filtered CSV",
                                  class="btn-success"))
          ),
          br(),
          DTOutput("raw_data")
        ),

        # ===================================================================
        # Tab 7 — Regression Models
        # In-app Poisson / Negative Binomial fit; mirrors, on a simpler grain,
        # the model in src/glm_inference.py (Chapter 4 of the thesis).
        # ===================================================================
        tabPanel("Regression",
          br(),
          div(class="section-header",
            h4("GLM for count data: Poisson vs Negative Binomial"),
            p("This tab fits the same model family used in",
              em("src/glm_inference.py"), " (Chapter 4 of the thesis)",
              " using R's", code("glm()"), " and", code("MASS::glm.nb()."),
              " The response variable is the", strong("number of donation events"),
              " per donor within the filtered scope (aggregated over all years).",
              " The Poisson GLM assumes equidispersion (Var = Mean); the Negative",
              " Binomial adds a dispersion parameter", em("θ"), " to accommodate",
              " Var > Mean. In the Python notebook, Var/Mean ≈ 1.83 for the full",
              " dataset, suggesting mild overdispersion — confirmed by the lower AIC",
              " of the NB model. Coefficients on the log scale; exp(β) = Incidence",
              " Rate Ratio (IRR).")
          ),
          br(),
          fluidRow(
            column(3,
              wellPanel(
                h4("Model configuration"),
                checkboxGroupInput("model_covariates","Covariates:",
                  choices  = c("Age"           = "age",
                               "Gender"        = "gender_label",
                               "Donation Type" = "donation_type"),
                  selected = c("age","gender_label","donation_type")),
                checkboxInput("model_interaction",
                              "Include Gender × Type interaction",
                              value = FALSE),
                hr(),
                helpText("Response = total donation events per donor in scope.",
                         " Poisson is the baseline; NB adds θ for overdispersion.")
              )
            ),
            column(9,
              fluidRow(
                column(6,
                       h3("Poisson",           style="color:#2c3e50; font-weight:bold;"),
                       uiOutput("poisson_metrics"),
                       h5("Coefficients",      style="font-weight:bold; margin-top:15px;"),
                       DTOutput("poisson_coefs")),
                column(6,
                       h3("Negative Binomial", style="color:#2980b9; font-weight:bold;"),
                       uiOutput("nb_metrics"),
                       h5("Coefficients",      style="font-weight:bold; margin-top:15px;"),
                       DTOutput("nb_coefs"))
              )
            )
          )
        ), # end Regression tab

        # ===================================================================
        # Tab 8 - Forecast 2026-2031
        # ===================================================================
        tabPanel("Forecast 2026–2031",
                 br(),
                 p("Six-year projection (damped-trend model, ",
                   tags$code("src/forecast.py"), ") generated by ",
                   tags$code("make export-legacy"),
                   ". Re-run that command to refresh these views."),
                 conditionalPanel(
                   condition = "output.forecast_loaded == false",
                   div(class = "alert alert-warning",
                       "Forecast files not found. Run ",
                       tags$code("make forecast export-legacy"),
                       " to generate them under ",
                       tags$code("data/FINAL/forecasts/"), ".")
                 ),
                 conditionalPanel(
                   condition = "output.forecast_loaded == true",
                   uiOutput("forecast_kpis"),
                   br(),
                   fluidRow(
                     column(4,
                       wellPanel(
                         h4("Calendar view"),
                         selectInput("fc_year", "Forecast year:",
                                     choices = 2026:2031, selected = 2026),
                         selectInput("fc_month", "Month:",
                                     choices = setNames(1:12, month.name),
                                     selected = 1),
                         hr(),
                         helpText("Daily expected donations from disaggregating the monthly forecast using the empirical weekday × month profile.")
                       )
                     ),
                     column(8, plotOutput("fc_calendar_plot", height = "440px"))
                   ),
                   br(),
                   fluidRow(
                     column(6, plotOutput("fc_monthly_plot", height = "320px")),
                     column(6, plotOutput("fc_type_plot", height = "320px"))
                   ),
                   br(),
                   fluidRow(
                     column(6, plotOutput("fc_donors_plot", height = "320px")),
                     column(6, plotOutput("fc_retention_plot", height = "320px"))
                   ),
                   br(),
                   h4("Monthly summary"),
                   p("Twelve rows, one per calendar month, for the",
                     em("Forecast year"), " selected above.",
                     style = "color:#666; margin-top:-8px;"),
                   DTOutput("fc_summary_table")
                 )
        ) # end Forecast tab
      ) # end tabsetPanel (main)
    ) # end mainPanel
  ) # end sidebarLayout
) # end fluidPage


# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {

  # ===========================================================================
  # 0. DATA LOAD (runs once on startup)
  # ===========================================================================
  # reactiveVal() caches the data so all downstream reactives share one copy.
  dataset <- reactiveVal(NULL)

  # Without the export every tab would stay silently empty: say why instead.
  output$data_missing <- renderUI({
    if (file.exists(data_path)) return(NULL)
    div(class = "alert alert-warning",
        strong("Data not found. "),
        "The dashboard reads ", tags$code("data/FINAL/dataframe_cleaned.csv"),
        ", which is generated by the pipeline. Run ",
        tags$code("./run_pipeline.sh --dashboard"), " (or ",
        tags$code(".\\run_pipeline.ps1 --dashboard"), " on Windows) from the",
        " repository root, then restart the app. See README.md.")
  })

  observe({
    req(file.exists(data_path))

    # read_csv is significantly faster than base read.csv on large CSVs.
    # show_col_types=FALSE suppresses the column-type message in the console.
    df <- suppressWarnings(
      readr::read_csv(data_path, show_col_types=FALSE, progress=FALSE))

    # ---- Date parsing -------------------------------------------------------
    # activity_date has mixed formats ("YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DD").
    # Parse as POSIXct first, then coerce to Date for day-level aggregations.
    df$activity_dt <- suppressWarnings(
      as.POSIXct(df$activity_date, format="%Y-%m-%d %H:%M:%S", tz="UTC"))
    fallback <- is.na(df$activity_dt) & !is.na(df$activity_date)
    df$activity_dt[fallback] <- suppressWarnings(
      as.POSIXct(df$activity_date[fallback], format="%Y-%m-%d", tz="UTC"))

    df$Date  <- as.Date(df$activity_dt)
    df$Year  <- year(df$Date)
    df$Month <- month(df$Date)
    # wday(..., week_start=1): Monday=1, …, Sunday=7 — used by calendar & weekday bar.
    df$WDay  <- wday(df$Date, week_start=1)

    # ---- Populate sidebar inputs --------------------------------------------
    valid_years <- sort(unique(df$Year[!is.na(df$Year)]), decreasing=TRUE)
    updateSelectInput(session, "year_filter",
                      choices  = c("All", as.character(valid_years)),
                      selected = as.character(valid_years[1]))

    valid_types <- sort(unique(df$donation_type[!is.na(df$donation_type)]))
    updatePickerInput(session, "type_filter",
                      choices=valid_types, selected=valid_types)

    # Snap age slider to real data bounds (clamped 18-100 to filter outliers).
    age_min <- max(18,  floor(min(df$age, na.rm=TRUE)))
    age_max <- min(100, ceiling(max(df$age, na.rm=TRUE)))
    updateSliderInput(session, "age_range",
                      min=age_min, max=age_max, value=c(age_min, age_max))

    # Column picker for the Data Table tab.
    default_cols <- c("donor_id","gender_label","age","donation_type",
                      "quantity_ml","collection_point","association_desc",
                      "activity_date")
    updatePickerInput(session, "dt_cols",
                      choices  = names(df),
                      selected = intersect(default_cols, names(df)))

    dataset(df)
  })


  # ===========================================================================
  # 1. GLOBAL FILTERED DATA
  # ===========================================================================
  # All tabs (except Young Donor Retention) read from here.
  filtered_data <- reactive({
    req(dataset(), input$year_filter)
    df <- dataset()

    if (input$year_filter != "All")
      df <- df %>% dplyr::filter(Year == as.numeric(input$year_filter))

    if (input$gender_filter != "All")
      df <- df %>% dplyr::filter(gender_label == input$gender_filter)

    # Multi-select type filter: empty selection → 0 rows (intentional).
    if (!is.null(input$type_filter) && length(input$type_filter) > 0)
      df <- df %>% dplyr::filter(donation_type %in% input$type_filter)
    else
      df <- df[0, ]

    # Age slider.
    ar <- input$age_range
    if (!is.null(ar))
      df <- df %>% dplyr::filter(!is.na(age), age >= ar[1], age <= ar[2])

    df
  })


  # ===========================================================================
  # 2. OVERVIEW KPIs + yearly bar
  # ===========================================================================
  output$kpi_donations <- renderText(scales::comma(nrow(filtered_data())))
  output$kpi_donors    <- renderText(scales::comma(dplyr::n_distinct(filtered_data()$donor_id)))
  output$kpi_avg_age   <- renderText({
    df <- filtered_data()
    if (nrow(df)==0) "-" else sprintf("%.1f", mean(df$age, na.rm=TRUE))
  })
  output$kpi_volume <- renderText({
    df <- filtered_data()
    if (nrow(df)==0) "-"
    else scales::comma(sum(df$quantity_ml, na.rm=TRUE)/1000, accuracy=0.1)
  })

  # Yearly bar — applies every filter EXCEPT year so context bars always show.
  output$overview_yearly <- renderPlot({
    req(dataset())
    df <- dataset()
    if (input$gender_filter != "All")
      df <- df %>% dplyr::filter(gender_label == input$gender_filter)
    if (!is.null(input$type_filter) && length(input$type_filter)>0)
      df <- df %>% dplyr::filter(donation_type %in% input$type_filter)
    ar <- input$age_range
    df <- df %>% dplyr::filter(!is.na(age), age >= ar[1], age <= ar[2])

    yearly <- df %>% dplyr::filter(!is.na(Year)) %>% dplyr::count(Year)
    yearly$highlight <- if (input$year_filter=="All") FALSE
                        else yearly$Year==as.numeric(input$year_filter)

    ggplot(yearly, aes(x=factor(Year), y=n, fill=highlight)) +
      geom_col() +
      scale_fill_manual(values=c(`FALSE`="steelblue",`TRUE`="firebrick"),
                        guide="none") +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Donations per Year (sidebar filters applied, Year filter excluded)",
           x="Year", y="Donations") +
      theme_thesis()
  })


  # ===========================================================================
  # 3. CALENDAR HEATMAP
  # ===========================================================================
  current_month_date <- reactiveVal(as.Date(NA))

  # Keep calendar anchor in sync with the year selector.
  observeEvent(input$year_filter, {
    req(input$year_filter, dataset())
    curr <- current_month_date()
    if (input$year_filter=="All") {
      if (is.na(curr)) {
        yrs <- unique(dataset()$Year); yrs <- yrs[!is.na(yrs)]
        current_month_date(as.Date(paste0(max(yrs),"-01-01")))
      }
    } else {
      tgt <- as.numeric(input$year_filter)
      if (is.na(curr) || year(curr)!=tgt)
        current_month_date(as.Date(paste0(tgt,"-01-01")))
    }
  })

  # Navigation helper: moves the calendar and keeps the year dropdown in sync.
  navigate_calendar <- function(new_date) {
    current_month_date(new_date)
    if (input$year_filter!="All" && year(new_date)!=as.numeric(input$year_filter))
      updateSelectInput(session,"year_filter", selected=as.character(year(new_date)))
  }
  observeEvent(input$prev_month, { req(!is.na(current_month_date()))
    navigate_calendar(current_month_date() %m-% months(1)) })
  observeEvent(input$next_month, { req(!is.na(current_month_date()))
    navigate_calendar(current_month_date() %m+% months(1)) })
  observeEvent(input$prev_year,  { req(!is.na(current_month_date()))
    navigate_calendar(current_month_date() %m-% years(1)) })
  observeEvent(input$next_year,  { req(!is.na(current_month_date()))
    navigate_calendar(current_month_date() %m+% years(1)) })

  output$month_year_text <- renderText({
    req(!is.na(current_month_date()))
    d <- current_month_date()
    paste(MONTH_LABELS_IT[month(d)], year(d))
  })

  output$calendar_plot <- renderPlot({
    req(filtered_data(), !is.na(current_month_date()))
    df         <- filtered_data()
    start_date <- current_month_date()
    end_date <- ceiling_date(start_date, "month") - days(1)
    
    if(nrow(df) == 0) {
      daily_counts <- data.frame(Date = as.Date(character()), Donations = numeric())
    } else {
      daily_counts <- df %>%
        filter(!is.na(Date), Date >= start_date, Date <= end_date) %>%
        group_by(Date) %>%
        summarise(Donations = n(), .groups = "drop")
    }
    
    # Generate full month grid
    all_days <- data.frame(Date = seq.Date(start_date, end_date, by="day"))
    
    cal_data <- left_join(all_days, daily_counts, by = "Date")
    cal_data$Donations[is.na(cal_data$Donations)] <- 0
    
    # Map numeric week day to standard English labels to prevent locale translation bugs
    eng_wdays <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    wd_int <- wday(cal_data$Date, week_start=1)
    
    # Calculate week of month mapping starting from Monday
    cal_data <- cal_data %>%
      mutate(WeekOfMonth = ceiling((day(Date) + wday(floor_date(Date, "month"), week_start=1) - 1) / 7))
    
    # Reverse weekday levels for standard calendar (Mon top, Sun bottom)
    cal_data$Weekday <- factor(eng_wdays[wd_int], levels = rev(eng_wdays))
    
    max_don <- max(cal_data$Donations, na.rm = TRUE)
    cal_data$TextColor <- "black"
    if (!is.infinite(max_don) && max_don > 0) {
      cal_data$TextColor <- ifelse(cal_data$Donations > max_don * 0.6, "white", "black")
    }
    
    ggplot(cal_data, aes(x = WeekOfMonth, y = Weekday, fill = Donations)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = day(Date), color = TextColor), size = 5, fontface = "bold", show.legend = FALSE) +
      scale_fill_gradient(low = "lightblue", high = "firebrick", na.value = "white") +
      scale_color_identity() +
      coord_equal(ratio=1) +
      labs(x=NULL, y=NULL, fill="Donations",
           subtitle=sprintf("Total in view: %s", scales::comma(sum(cal_data$Donations)))) +
      theme_thesis() +
      theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),
            axis.text.y=element_text(size=12,face="bold"),
            panel.grid=element_blank())
  })


  # ===========================================================================
  # 4. AGE (ETA) — sub-tab 3a: four standard plots
  # ===========================================================================

  # 4.1 Histogram with mean line
  output$age_histogram <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    mean_age <- mean(df$age, na.rm=TRUE)
    ggplot(df, aes(x=age)) +
      geom_histogram(binwidth=2, fill="steelblue", color="white") +
      geom_vline(xintercept=mean_age, color="firebrick", linewidth=1.1,
                 linetype="dashed") +
      annotate("text", x=mean_age+1, y=Inf,
               label=sprintf("Mean: %.1f", mean_age),
               hjust=0, vjust=1.5, color="firebrick", size=3.8) +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Age Distribution", x="Age (years)", y="Donations") +
      theme_thesis()
  })

  # 4.2 Age by gender — boxplot
  output$age_by_gender <- renderPlot({
    df <- filtered_data() %>%
      dplyr::filter(!is.na(gender_label), gender_label!="")
    if (nrow(df)==0) return(NULL)
    ggplot(df, aes(x=gender_label, y=age, fill=gender_label)) +
      geom_boxplot(alpha=0.85, outlier.alpha=0.2) +
      scale_fill_brewer(palette="Set1", guide="none") +
      labs(title="Age by Gender", x=NULL, y="Age") +
      theme_thesis()
  })

  # 4.3 Age by donation type — boxplot
  output$age_by_type <- renderPlot({
    df <- filtered_data() %>%
      dplyr::filter(!is.na(donation_type), donation_type!="")
    if (nrow(df)==0) return(NULL)
    ggplot(df, aes(x=donation_type, y=age, fill=donation_type)) +
      geom_boxplot(alpha=0.85, outlier.alpha=0.2) +
      scale_fill_brewer(palette="Set2", guide="none") +
      labs(title="Age by Donation Type", x=NULL, y="Age") +
      theme_thesis() +
      theme(axis.text.x=element_text(angle=30,hjust=1))
  })

  # 4.4 Mean age by birth cohort (decade) over years
  output$age_by_cohort <- renderPlot({
    df <- filtered_data() %>%
      dplyr::filter(!is.na(birth_cohort), !is.na(Year), !is.na(age))
    if (nrow(df)==0) return(NULL)
    df$cohort_decade <- floor(df$birth_cohort/10)*10
    summary_df <- df %>%
      dplyr::group_by(Year, cohort_decade) %>%
      dplyr::summarise(mean_age=mean(age,na.rm=TRUE), n=dplyr::n(),
                       .groups="drop") %>%
      dplyr::filter(n>=20)
    ggplot(summary_df,
           aes(x=Year, y=mean_age, color=factor(cohort_decade),
               group=cohort_decade)) +
      geom_line(linewidth=1) + geom_point() +
      labs(title="Mean Age by Birth Cohort (decade) over Years",
           x="Year", y="Mean age", color="Cohort (decade)") +
      theme_thesis()
  })


  # ===========================================================================
  # 5. AGE (ETA) — sub-tab 3b: Young Donor Retention
  # ===========================================================================
  # ---- 5.0 Per-donor summary (gender-filtered ONLY — no year/age/type) -------
  # Computing this separately from filtered_data() is deliberate: applying the
  # year filter would cut retention_years = last_year - first_year at the
  # selected year boundary, making all donors look like one-time donors.
  donor_summary <- reactive({
    req(dataset())
    df <- dataset()

    # Apply gender filter so M/F comparisons work, but NO year filter.
    if (input$gender_filter != "All")
      df <- df %>% dplyr::filter(gender_label == input$gender_filter)

    df %>%
      dplyr::filter(!is.na(donor_id), !is.na(donation_year), !is.na(age)) %>%
      dplyr::group_by(donor_id) %>%
      dplyr::summarise(
        first_year      = min(donation_year, na.rm=TRUE),
        last_year       = max(donation_year, na.rm=TRUE),
        first_age       = min(age,           na.rm=TRUE),
        last_age        = max(age,            na.rm=TRUE),
        total_donations = dplyr::n(),
        gender_label    = dplyr::first(gender_label),
        .groups         = "drop"
      ) %>%
      dplyr::mutate(retention_years = last_year - first_year)
  })

  # ---- 5.1 Young donors filtered by the in-tab slider ----------------------
  young_donors_rv <- reactive({
    req(donor_summary(), input$young_age_range)
    ar <- input$young_age_range
    donor_summary() %>%
      dplyr::filter(first_age >= ar[1], first_age <= ar[2])
  })

  # ---- 5.2 KPI outputs ------------------------------------------------------
  output$kpi_young_n    <- renderText(scales::comma(nrow(young_donors_rv())))
  output$kpi_one_time   <- renderText({
    yd <- young_donors_rv(); if (nrow(yd)==0) return("-")
    scales::percent(mean(yd$retention_years==0), accuracy=0.1)
  })
  output$kpi_retained_1 <- renderText({
    yd <- young_donors_rv(); if (nrow(yd)==0) return("-")
    scales::percent(mean(yd$retention_years>=1), accuracy=0.1)
  })
  output$kpi_retained_5 <- renderText({
    yd <- young_donors_rv(); if (nrow(yd)==0) return("-")
    scales::percent(mean(yd$retention_years>=5), accuracy=0.1)
  })

  # ---- 5.3 Retention bar chart (replicates 'Option B' in the notebook) ------
  output$retention_bar <- renderPlot({
    yd <- young_donors_rv(); if (nrow(yd)==0) return(NULL)

    ret_dist <- yd %>%
      dplyr::count(retention_years) %>%
      dplyr::arrange(retention_years)

    ggplot(ret_dist, aes(x=factor(retention_years), y=n)) +
      geom_col(fill="steelblue", color="white") +
      geom_text(aes(label=scales::comma(n)), vjust=-0.4, size=3.4) +
      scale_y_continuous(labels=scales::comma,
                         expand=expansion(mult=c(0,0.12))) +
      labs(
        title    = "Retention Period Distribution",
        subtitle = sprintf("Young donors (first age %d-%d)  |  N = %s",
                           input$young_age_range[1], input$young_age_range[2],
                           scales::comma(nrow(yd))),
        x = "Years between first and last recorded donation",
        y = "Number of donors"
      ) +
      theme_thesis()
  })

  # ---- 5.4 Retention pie: one-time vs retained (replicates 'Option A') ------
  output$retention_pie <- renderPlot({
    yd <- young_donors_rv(); if (nrow(yd)==0) return(NULL)

    pie_df <- yd %>%
      dplyr::mutate(
        status = ifelse(retention_years==0, "One-time", "Retained \u22651 yr")) %>%
      dplyr::count(status) %>%
      dplyr::mutate(pct = n/sum(n),
                    lbl = paste0(status,"\n",scales::percent(pct,accuracy=0.1)))

    ggplot(pie_df, aes(x="", y=n, fill=status)) +
      geom_bar(stat="identity", width=1, color="white") +
      coord_polar("y", start=0) +
      geom_text(aes(label=lbl),
                position=position_stack(vjust=0.5),
                color="white", fontface="bold", size=4.5) +
      scale_fill_manual(
        values=c("One-time"="firebrick","Retained \u22651 yr"="steelblue")) +
      labs(title="One-time vs Retained", fill=NULL) +
      theme_void() +
      theme(plot.title=element_text(hjust=0.5,face="bold"),
            legend.position="none")
  })

  # ---- 5.5 Cumulative retention curve ---------------------------------------
  # Equivalent to a simplified Kaplan-Meier: for each N, show the % of young
  # donors whose retention_years >= N.
  output$retention_cumulative <- renderPlot({
    yd <- young_donors_rv(); if (nrow(yd)==0) return(NULL)

    max_yrs <- max(yd$retention_years, na.rm=TRUE)
    cumul <- data.frame(
      years = 0:max_yrs,
      pct   = sapply(0:max_yrs, function(y) mean(yd$retention_years >= y))
    )

    ggplot(cumul, aes(x=years, y=pct)) +
      geom_area(fill="steelblue", alpha=0.15) +
      geom_line(color="steelblue",  linewidth=1.2) +
      geom_point(color="steelblue", size=2.5) +
      scale_y_continuous(labels=scales::percent, limits=c(0,1)) +
      scale_x_continuous(breaks=0:max_yrs) +
      labs(
        title    = "Cumulative Retention Curve",
        subtitle = "Proportion of young donors who donated for at least N years",
        x = "Years since first donation (N)",
        y = "Proportion active \u2265 N years"
      ) +
      theme_thesis()
  })

  # ---- 5.6 Retention distribution table ------------------------------------
  output$retention_table <- renderDT({
    yd <- young_donors_rv()
    if (nrow(yd)==0) return(datatable(data.frame()))

    tbl <- yd %>%
      dplyr::count(retention_years, name="donors") %>%
      dplyr::arrange(retention_years) %>%
      dplyr::mutate(
        pct_of_total       = scales::percent(donors/nrow(yd), accuracy=0.1),
        # "% active >= N years" = cumulative retention rate at year N
        cumul_pct_retained = scales::percent(
          sapply(retention_years,
                 function(y) mean(yd$retention_years >= y)),
          accuracy = 0.1)
      )
    colnames(tbl) <- c("Retention Years","Donors",
                        "% of Total","% Active \u2265 N Years")
    datatable(tbl, rownames=FALSE,
              options=list(pageLength=20, dom="t", scrollX=TRUE))
  })


  # ===========================================================================
  # 6. AGE (ETA) — sub-tab 3c: Frequency & Volume by Age Group
  # ===========================================================================

  # ---- 6.1 Avg yearly donations per 5-year age band -------------------------
  # Mirrors the per-age-band frequency analysis in src/eda.py.
  output$age_freq_bar <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)

    # Bin age into 5-year bands (18-22, 23-27, …, 68-72).
    df$age_group <- cut(df$age,
                        breaks = seq(15, 75, by=5),
                        labels = paste0(seq(15,70,by=5),"-",seq(19,74,by=5)),
                        right  = FALSE)

    freq_df <- df %>%
      dplyr::filter(!is.na(age_group), !is.na(donation_year)) %>%
      # Count events per donor per year …
      dplyr::group_by(donor_id, donation_year, age_group) %>%
      dplyr::summarise(n_don=dplyr::n(), .groups="drop") %>%
      # … then average across donors in the same age band.
      dplyr::group_by(age_group) %>%
      dplyr::summarise(avg_donations=mean(n_don), .groups="drop")

    ggplot(freq_df, aes(x=age_group, y=avg_donations)) +
      geom_col(fill="darkorange", color="white") +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Avg Yearly Donations by Age Group",
           x="Age band (years)", y="Avg donations per donor-year") +
      theme_thesis() +
      theme(axis.text.x=element_text(angle=45, hjust=1))
  })

  # ---- 6.2 Total donated volume vs donor age (scatter + LOESS) --------------
  output$age_totalvol_scatter <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)

    # Aggregate: total volume (L) and max observed age, per donor.
    donor_vol <- df %>%
      dplyr::filter(!is.na(quantity_ml)) %>%
      dplyr::group_by(donor_id) %>%
      dplyr::summarise(
        total_vol_L = sum(quantity_ml, na.rm=TRUE)/1000,
        age         = max(age, na.rm=TRUE),
        .groups     = "drop"
      )

    # Sample up to 5,000 donors for rendering performance.
    if (nrow(donor_vol) > 5000)
      donor_vol <- donor_vol[sample(nrow(donor_vol), 5000), ]

    ggplot(donor_vol, aes(x=age, y=total_vol_L)) +
      geom_point(alpha=0.18, color="steelblue", size=1) +
      geom_smooth(method="loess", se=TRUE,
                  color="firebrick", fill="firebrick",
                  alpha=0.2, linewidth=1.2) +
      scale_y_continuous(labels=scales::comma) +
      labs(title    = "Total Donated Volume vs. Donor Age",
           subtitle = "Sample \u2264 5,000 donors | LOESS trend \u00b1 95% CI",
           x = "Donor age (years)",
           y = "Total volume donated (L)") +
      theme_thesis()
  })


  # ===========================================================================
  # 7. TIME SERIES
  # ===========================================================================
  output$ts_yearly <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    yearly <- df %>% dplyr::filter(!is.na(Year)) %>% dplyr::count(Year)
    ggplot(yearly, aes(x=Year, y=n)) +
      geom_line(color="steelblue", linewidth=1.1) + geom_point(color="steelblue") +
      scale_y_continuous(labels=scales::comma) +
      scale_x_continuous(breaks=yearly$Year) +
      labs(title="Donations per Year", x="Year", y="Donations") +
      theme_thesis()
  })

  output$ts_monthly <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    monthly <- df %>%
      dplyr::filter(!is.na(Month), !is.na(Year)) %>%
      dplyr::count(Year, Month) %>%
      dplyr::group_by(Month) %>%
      dplyr::summarise(avg=mean(n), .groups="drop")
    ggplot(monthly, aes(x=factor(Month), y=avg)) +
      geom_col(fill="darkorange") +
      scale_x_discrete(labels=substr(MONTH_LABELS_IT,1,3)) +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Avg Donations per Month (seasonality)",
           x=NULL, y="Avg donations / year") +
      theme_thesis()
  })

  output$ts_weekday <- renderPlot({
    df <- filtered_data() %>% dplyr::filter(!is.na(WDay))
    if (nrow(df)==0) return(NULL)
    wkly <- df %>% dplyr::count(WDay)
    wkly$Weekday <- factor(WEEKDAY_LABELS_EN[wkly$WDay], levels=WEEKDAY_LABELS_EN)
    ggplot(wkly, aes(x=Weekday, y=n)) +
      geom_col(fill="seagreen") +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Donations by Weekday", x=NULL, y="Donations") +
      theme_thesis()
  })


  # ===========================================================================
  # 8. UNIVARIATE (pies + bars)
  # ===========================================================================
  # Shared pie-chart factory: builds a ggplot pie with percentage labels.
  make_pie <- function(df, col, title, palette="Set1") {
    counts <- df %>%
      dplyr::count(.data[[col]]) %>%
      dplyr::filter(!is.na(.data[[col]]), .data[[col]]!="") %>%
      dplyr::mutate(pct=n/sum(n),
                    lbl=scales::percent(pct, accuracy=0.1))
    if (nrow(counts)==0) return(NULL)
    ggplot(counts, aes(x="", y=n, fill=.data[[col]])) +
      geom_bar(stat="identity", width=1, color="white") +
      coord_polar("y", start=0) +
      geom_text(aes(label=lbl),
                position=position_stack(vjust=0.5),
                color="white", fontface="bold") +
      scale_fill_brewer(palette=palette) +
      labs(title=title, fill=NULL) +
      theme_void() +
      theme(plot.title=element_text(hjust=0.5,face="bold"))
  }

  output$gender_pie     <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    make_pie(df,"gender_label","Gender Distribution","Set1") })
  output$type_pie       <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    make_pie(df,"donation_type","Donation Type Distribution","Set2") })

  output$collection_bar <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    counts <- df %>% dplyr::count(collection_point) %>%
      dplyr::filter(!is.na(collection_point),collection_point!="") %>%
      dplyr::arrange(n) %>%
      dplyr::mutate(collection_point=factor(collection_point,levels=collection_point))
    ggplot(counts, aes(x=collection_point,y=n)) +
      geom_col(fill="steelblue") + coord_flip() +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Donations by Collection Point",x=NULL,y="Donations") +
      theme_thesis()
  })

  output$class_bar <- renderPlot({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    counts <- df %>% dplyr::count(donor_class) %>%
      dplyr::filter(!is.na(donor_class), donor_class!="")
    ggplot(counts, aes(x=donor_class,y=n)) +
      geom_col(fill="forestgreen") +
      scale_y_continuous(labels=scales::comma) +
      labs(title="Donations by Donor Class",x="Donor class",y="Donations") +
      theme_thesis()
  })


  # ===========================================================================
  # 9. DATA TABLE — sortable / filterable / downloadable
  # ===========================================================================
  dt_visible <- reactive({
    req(filtered_data(), input$dt_cols)
    df   <- filtered_data()
    cols <- intersect(input$dt_cols, names(df))
    if (length(cols)==0) cols <- names(df)[1:min(5,ncol(df))]
    df[, cols, drop=FALSE]
  })

  output$raw_data <- renderDT(
    datatable(dt_visible(), rownames=FALSE, filter="top",
              options=list(pageLength=15, scrollX=TRUE, dom="lfrtip"))
  )

  output$dt_download <- downloadHandler(
    filename = function()
      sprintf("donations_filtered_%s.csv", format(Sys.Date(),"%Y%m%d")),
    content  = function(file) readr::write_csv(dt_visible(), file)
  )


  # ===========================================================================
  # 10. REGRESSION — Poisson & Negative Binomial
  # In-app fit; mirrors, on a simpler grain, src/glm_inference.py.
  # ===========================================================================
  # Aggregate filtered rows to one row per (donor × gender × type).
  # Response = total_donations within the filter scope.
  aggregated_data <- reactive({
    df <- filtered_data(); if (nrow(df)==0) return(NULL)
    df_agg <- df %>%
      dplyr::group_by(donor_id, gender_label, donation_type) %>%
      dplyr::summarise(total_donations=dplyr::n(),
                       age=max(age,na.rm=TRUE), .groups="drop") %>%
      dplyr::filter(!is.na(gender_label), gender_label!="",
                    !is.na(donation_type), donation_type!="",
                    !is.na(age), is.finite(age))
    df_agg$gender_label  <- as.factor(df_agg$gender_label)
    df_agg$donation_type <- as.factor(df_agg$donation_type)
    df_agg
  })

  # Build model formula from checkbox selection + optional interaction term.
  model_formula <- reactive({
    covs <- input$model_covariates
    rhs  <- if (length(covs)==0) "1" else paste(covs, collapse=" + ")
    if (isTRUE(input$model_interaction) &&
        all(c("gender_label","donation_type") %in% covs))
      rhs <- paste(rhs, "+ gender_label:donation_type")
    as.formula(paste("total_donations ~", rhs))
  })

  poisson_model <- reactive({
    df_agg <- aggregated_data()
    if (is.null(df_agg) || nrow(df_agg)<5) return(NULL)
    tryCatch(glm(model_formula(), data=df_agg, family="poisson"),
             error=function(e) NULL)
  })
  nb_model <- reactive({
    df_agg <- aggregated_data()
    if (is.null(df_agg) || nrow(df_agg)<5) return(NULL)
    tryCatch(MASS::glm.nb(model_formula(), data=df_agg),
             error=function(e) NULL)
  })

  # Format coefficient table for DT display.
  extract_coefs <- function(model) {
    if (is.null(model)) return(data.frame())
    ct <- as.data.frame(summary(model)$coefficients)
    ct$Predictor <- rownames(ct); rownames(ct) <- NULL
    ct[,1:3]     <- lapply(ct[,1:3], round, 4)
    colnames(ct) <- c("Estimate","Std. Error","z/t value","P-value","Predictor")
    # exp(beta) = Incidence Rate Ratio, the scale the thesis reports in
    # Chapter 4's IRR table; without it the header text below promises a
    # quantity the table never shows.
    ct$IRR       <- round(exp(ct$Estimate), 3)
    ct           <- ct[, c("Predictor","Estimate","IRR","Std. Error","z/t value","P-value")]
    ct$`P-value` <- format.pval(ct$`P-value`, digits=3, eps=0.001)
    ct
  }
  
  output$poisson_metrics <- renderUI({
    model <- poisson_model()
    if(is.null(model)) return(tags$div(class="alert alert-warning", "Not enough data or error fitting model."))
    
    tags$div(style = "background-color: #f7f9f9; padding: 15px; border-radius: 8px; border-left: 5px solid #2c3e50; margin-bottom: 10px; min-height: 165px;",
             h4(style="margin-top:0;", "Key Metrics"),
             tags$ul(style="list-style-type: none; padding-left: 0; font-size: 1.1em;",
                     tags$li(tags$strong("AIC: "), round(AIC(model), 1)),
                     tags$li(tags$strong("Residual Deviance: "), round(model$deviance, 1)),
                     tags$li(tags$strong("Degrees of Freedom: "), model$df.residual)
             )
    )
  })
  
  output$nb_metrics <- renderUI({
    model <- nb_model()
    if(is.null(model)) return(tags$div(class="alert alert-warning", "Not enough data or error fitting model."))
    
    tags$div(style = "background-color: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 5px solid #2980b9; margin-bottom: 10px; min-height: 165px;",
             h4(style="margin-top:0;", "Key Metrics"),
             tags$ul(style="list-style-type: none; padding-left: 0; font-size: 1.1em;",
                     tags$li(tags$strong("AIC: "), round(AIC(model), 1)),
                     tags$li(tags$strong("Residual Deviance: "), round(model$deviance, 1)),
                     tags$li(tags$strong("Degrees of Freedom: "), model$df.residual),
                     tags$li(tags$strong("Theta (Dispersion): "), round(model$theta, 3), " (SE: ", round(model$SE.theta, 3), ")")
             )
    )
  })
  
  output$poisson_coefs <- renderDT({
    req(poisson_model())
    coefs <- extract_coefs(poisson_model())
    datatable(coefs, options = list(pageLength = 10, dom = 't', scrollX = TRUE), rownames = FALSE) %>%
      formatStyle('P-value', color = styleInterval(c(0.05), c('green', 'black')), fontWeight = styleInterval(c(0.05), c('bold', 'normal')))
  })
  
  output$nb_coefs <- renderDT({
    req(nb_model())
    coefs <- extract_coefs(nb_model())
    datatable(coefs, options = list(pageLength = 10, dom = 't', scrollX = TRUE), rownames = FALSE) %>%
      formatStyle('P-value', color = styleInterval(c(0.05), c('green', 'black')), fontWeight = styleInterval(c(0.05), c('bold', 'normal')))
  })

  # ===== 5. Forecast 2026-2031 =====

  output$forecast_loaded <- reactive({ forecast_available })
  outputOptions(output, "forecast_loaded", suspendWhenHidden = FALSE)

  fc_data <- reactive({
    req(forecast_available)
    list(
      daily          = read.csv(fc_paths$daily,   stringsAsFactors = FALSE) %>% mutate(date = as.Date(date)),
      monthly        = read.csv(fc_paths$monthly, stringsAsFactors = FALSE) %>% mutate(month_start = as.Date(month_start)),
      type           = read.csv(fc_paths$type,    stringsAsFactors = FALSE),
      newd           = read.csv(fc_paths$newd,    stringsAsFactors = FALSE),
      ret            = read.csv(fc_paths$ret,     stringsAsFactors = FALSE),
      donors_monthly = read.csv(fc_paths$donors_monthly, stringsAsFactors = FALSE),
      summary        = read.csv(fc_paths$summary, stringsAsFactors = FALSE)
    )
  })

  output$forecast_kpis <- renderUI({
    req(fc_data())
    s <- fc_data()$summary
    total_5y <- sum(s$forecast_donations)
    avg_donors <- round(mean(s$expected_total_donors))
    avg_new <- round(mean(s$expected_new_donors))
    avg_ret <- round(mean(s$retention_rate_pct), 1)

    kpi_box <- function(label, value, color) {
      div(style = sprintf("background-color:%s; color:white; padding:14px; border-radius:8px; text-align:center;", color),
          h3(style="margin:0; font-weight:bold;", value),
          tags$small(label))
    }
    fluidRow(
      column(3, kpi_box("Total donations 2026-2031", format(total_5y, big.mark = ","), "#34495e")),
      column(3, kpi_box("Avg unique donors / yr",   format(avg_donors, big.mark = ","), "#2980b9")),
      column(3, kpi_box("Avg new donors / yr",      format(avg_new, big.mark = ","), "#e67e22")),
      column(3, kpi_box("Avg retention rate",       paste0(avg_ret, "%"), "#27ae60"))
    )
  })

  output$fc_calendar_plot <- renderPlot({
    req(fc_data())
    yr <- as.integer(input$fc_year)
    mo <- as.integer(input$fc_month)
    daily <- fc_data()$daily
    sub <- daily %>%
      filter(year(date) == yr, month(date) == mo)
    if(nrow(sub) == 0) return(NULL)

    eng_wdays <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    sub <- sub %>%
      mutate(WeekOfMonth = ceiling((day(date) + wday(floor_date(date, "month"), week_start = 1) - 1) / 7),
             Weekday = factor(eng_wdays[wday(date, week_start = 1)], levels = rev(eng_wdays)),
             label = day(date))

    max_d <- max(sub$donations, na.rm = TRUE)
    sub$TextColor <- ifelse(sub$donations > max_d * 0.6, "white", "black")

    ggplot(sub, aes(x = WeekOfMonth, y = Weekday, fill = donations)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = label, color = TextColor), size = 5, fontface = "bold", show.legend = FALSE) +
      scale_fill_gradient(low = "#fff5d6", high = "#c0392b", na.value = "white") +
      scale_color_identity() +
      coord_equal(ratio = 1) +
      theme_minimal() +
      theme(panel.grid = element_blank(),
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.y = element_text(size = 12, face = "bold"),
            plot.title = element_text(face = "bold", hjust = 0.5),
            legend.position = "right") +
      labs(title = paste0("Forecast donations — ", month.name[mo], " ", yr),
           x = NULL, y = NULL, fill = "Donations")
  })

  output$fc_monthly_plot <- renderPlot({
    req(fc_data())
    df_m <- fc_data()$monthly
    # Use the corrected monthly history (2020 & 2023 fixed) produced by the notebook.
    # Fall back to the raw dataset only if the corrected file is unavailable.
    if(file.exists(fc_paths$monthly_hist)) {
      hist_monthly <- read.csv(fc_paths$monthly_hist, stringsAsFactors = FALSE) %>%
        mutate(month_start = as.Date(month_start))
    } else {
      df_h <- dataset()
      if(is.null(df_h)) return(NULL)
      hist_monthly <- df_h %>%
        filter(!is.na(Date)) %>%
        mutate(month_start = floor_date(Date, "month")) %>%
        group_by(month_start) %>%
        summarise(donations = n(), .groups = "drop")
    }
    ggplot() +
      geom_line(data = hist_monthly, aes(x = month_start, y = donations, color = "Observed"), linewidth = 0.4) +
      geom_ribbon(data = df_m, aes(x = month_start, ymin = lower_90, ymax = upper_90), fill = "#c0392b", alpha = 0.2) +
      geom_line(data = df_m, aes(x = month_start, y = forecast, color = "Forecast"), linewidth = 0.7) +
      scale_color_manual(values = c("Observed" = "#34495e", "Forecast" = "#c0392b")) +
      labs(title = "Monthly donations: history & forecast",
           x = "Month", y = "Donations / month", color = NULL) +
      theme_minimal() +
      theme(plot.title = element_text(face = "bold", hjust = 0.5),
            legend.position = "top")
  })

  output$fc_type_plot <- renderPlot({
    req(fc_data())
    tdf <- fc_data()$type
    ggplot(tdf, aes(x = factor(year), y = donations, fill = donation_type)) +
      geom_bar(stat = "identity") +
      scale_fill_brewer(palette = "Set2") +
      labs(title = "Forecast donations by type",
           x = "Year", y = "Donations", fill = "Type") +
      theme_minimal() +
      theme(plot.title = element_text(face = "bold", hjust = 0.5),
            legend.position = "right")
  })

  output$fc_donors_plot <- renderPlot({
    req(fc_data())
    # One bar per calendar month (Jan..Dec), not per (year, month): the
    # underlying yearly totals are held flat across 2026-2031 (last-observed
    # average, no evidence of a trend — see retention_rate below), so the
    # same 12-month profile would just repeat six times. The monthly split
    # applies the same empirical seasonal profile used for the volume
    # forecast to that one flat annual total.
    mdf <- fc_data()$donors_monthly
    long_df <- data.frame(
      month = factor(rep(month.abb[mdf$month], 2), levels = month.abb),
      type = factor(rep(c("Returning", "New"), each = nrow(mdf)),
                    levels = c("Returning", "New")),
      donors = c(mdf$expected_returning_donors, mdf$expected_new_donors)
    )
    ggplot(long_df, aes(x = month, y = donors, fill = type)) +
      geom_col() +
      scale_fill_manual(values = c("Returning" = "#2980b9", "New" = "#e67e22")) +
      labs(title = "Expected donors by month",
           subtitle = "Average calendar-month profile, flat across 2026-2031",
           x = NULL, y = "Donors", fill = NULL) +
      theme_minimal() +
      theme(plot.title = element_text(face = "bold", hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5, color = "grey35"),
            legend.position = "top")
  })

  output$fc_retention_plot <- renderPlot({
    req(fc_data())
    rdf <- fc_data()$ret
    ggplot(rdf, aes(x = factor(year), y = retention_rate * 100, group = 1)) +
      geom_line(color = "#c0392b", linewidth = 0.9) +
      geom_point(color = "#c0392b", size = 3) +
      geom_text(aes(label = paste0(round(retention_rate * 100, 1), "%")),
                vjust = -1, fontface = "bold", size = 3.6) +
      labs(title = "Forecast year-over-year retention rate",
           x = "Year", y = "Retention (%)") +
      theme_minimal() +
      theme(plot.title = element_text(face = "bold", hjust = 0.5)) +
      expand_limits(y = c(min(rdf$retention_rate * 100) - 2,
                          max(rdf$retention_rate * 100) + 4))
  })

  output$fc_summary_table <- renderDT({
    req(fc_data(), input$fc_year)
    yr <- as.integer(input$fc_year)
    # Forecast volume varies by year (the damped-trend model's declining
    # total), so filtering the 72-row monthly forecast to one year already
    # shows the year-to-year difference the table is meant to surface.
    # Expected donors don't: that split is a flat annual total (see the
    # Retention rate discussion) spread over the same 12-month seasonal
    # profile every year, so it repeats unchanged whichever year is picked.
    m <- fc_data()$monthly %>%
      mutate(month = month(month_start)) %>%
      filter(year(month_start) == yr) %>%
      arrange(month)
    tbl <- m %>%
      dplyr::select(month, forecast, lower_90, upper_90) %>%
      left_join(fc_data()$donors_monthly, by = "month") %>%
      transmute(
        Month = month.name[month],
        `Forecast donations` = format(round(forecast), big.mark = ","),
        `Lower 90%` = format(round(lower_90), big.mark = ","),
        `Upper 90%` = format(round(upper_90), big.mark = ","),
        `Expected new donors` = format(expected_new_donors, big.mark = ","),
        `Expected returning donors` = format(expected_returning_donors, big.mark = ",")
      )
    datatable(tbl, rownames = FALSE, options = list(dom = "t", pageLength = 12))
  })
}

# =============================================================================
# Launch
# =============================================================================
shinyApp(ui, server)
