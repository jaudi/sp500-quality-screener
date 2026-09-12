# Screener Pipeline

Python screeners that run in GitHub Actions and write JSON reports into this repo.
The Next.js portal at financeplots.com reads those JSONs live over
`raw.githubusercontent.com`.

**There is no server and no database — the repo is the database.** A run's only
output is a committed file under `data/`.

## Layout

- `common.py` — everything shared: technical indicators, both filters, the
  web-search tool, the Claude agents, and both pipeline runners
- `valuation.py` — the reverse-DCF stage. Pure arithmetic and yfinance, no
  Claude, so it can be tested without spending tokens.
- `screener.py` / `screener_ibex35.py` / `screener_nasdaq100.py` — thin entry
  points. Each one only knows how to source its universe and which runner to
  call, then goes into `common.py`.
- `data/latest-report.json` · `latest-report-ibex35.json` · `latest-report-nasdaq100.json`

To add an index, write a new entry point that produces a ticker list and calls
`run_pipeline` (quality) or `run_pipeline_crecimiento` (growth). Don't fork
`common.py` — everything behind the filter is already shared by both runners.

## Schedule

| Workflow | Cron | Entry point | Screen |
|---|---|---|---|
| Weekly Quality Screener | Mon 06:00 UTC | `screener.py` | quality |
| Weekly IBEX 35 | Tue 06:00 UTC | `screener_ibex35.py` | quality |
| Weekly Nasdaq-100 | Wed 06:00 UTC | `screener_nasdaq100.py` | growth |

All three also accept `workflow_dispatch`, so you can trigger a run by hand:

```
gh workflow run weekly-screener.yml -R jaudi/sp500-quality-screener
```

Each workflow commits its report back to `main` as a `chore:` commit. Expect
origin to be ahead of your local clone after a run.

## The two screens

There are two filters, not one, and which one an index gets is a judgement about
the index.

**Quality** (`filtrar_acciones_calidad`) — ROE > 20%, P/E < 20, Debt/Equity <
100%, RSI(14) > 30, price > MA50. The S&P run adds ROA > 12% for six criteria;
the IBEX run passes `roa_minimo=None` and applies five, because the smaller
universe leaves too few names otherwise. The count is derived, not hardcoded —
see `num_filtros`.

**Growth** (`filtrar_acciones_crecimiento`) — revenue growth > 10%, earnings
growth > 10%, positive free cash flow, price > MA50, MA50 > MA200, 6-month return
> 0, RSI(14) > 40. Seven criteria, no valuation or profitability filter at all.

Why the Nasdaq-100 gets the growth screen: a P/E < 20 filter rejects almost the
entire index, and what it lets through is the least representative of what the
index is. A low multiple there does not signal quality — it signals that the
market has stopped expecting growth, which is the opposite of what the screen is
looking for. Positive free cash flow is the only quality guardrail kept, and it
does double duty: it also happens to be what the valuation stage needs, since a
negative latest FCF makes the reverse DCF fail anyway.

Things that will bite you in the growth screen:

- **`earningsGrowth` is empty on a good number of names.** `earningsQuarterlyGrowth`
  is the same figure by another route and is used as the fallback. The `is None`
  check has to stay explicit: a growth rate of exactly 0.0 is data, not a gap,
  and an `or` would silently treat it as missing.
- **`score` is a within-cohort percentile rank, not a grade.** 100 means best of
  what passed this week. It is built from ranks rather than raw values on
  purpose: one company coming off a cyclical trough with +1,300% earnings growth
  would otherwise dominate the average and turn three columns into one wearing a
  disguise. The JSON says this in `criteria.score`, and the agent prompt says it
  again, because a 0-100 number next to a ticker reads as a grade to everyone.
- **Momentum needs 14 months of prices, not 4.** `calcular_indicadores_tecnicos`
  takes `con_tendencia_larga=True` for the MA200 and the 6/12-month returns. The
  quality screens deliberately don't pay for that longer download across 500
  tickers.
- **The universe does not come from Wikipedia.** The components table was removed
  from the Nasdaq-100 article — there is no ~100-row table left to scrape there.
  Slickcharts is the live source; if it fails, `NASDAQ100_FALLBACK` in
  `screener_nasdaq100.py` is a pinned list so the week is never lost to someone
  else's HTML change. Which one was used lands in the JSON as `universe_source`,
  not just in the Actions log, because a stale universe is invisible otherwise.

## The Claude agent

`generar_informe` runs a manual agentic loop against `claude-sonnet-5` with a
single client tool, `buscar_noticias_web` (DuckDuckGo via `ddgs`). It is a
manual loop rather than the SDK tool runner to avoid a beta dependency in an
unattended weekly job.

Things that will bite you:

- **`MAX_ITERACIONES_AGENTE` is a cost ceiling, not a formality.** This runs
  unattended and is billed per token. Groq's free tier forgave an unbounded
  loop; this does not.
- **Echo back the whole `response.content`, not just the text.** Adaptive
  thinking blocks have to survive between turns.
- **Every `tool_use` block needs a matching `tool_result`,** including unknown
  tools — otherwise the next request fails on an orphaned `tool_use_id`. Return
  all results in a single user message, or the model stops making parallel calls.
- **The loop itself lives in `_bucle_agentico`,** shared by the quality and
  growth reports. They ask different questions over the same mechanics; the
  three traps above are exactly why that mechanic is written once.
- If the report fails, `run_pipeline` still writes the screening results without
  it. Losing the commentary should never lose the week's data.

## The valuation stage (reverse DCF)

Runs after the screening report, on the names that passed. Two-stage DCF on
levered free cash flow: 10 explicit years plus a Gordon terminal at 2.5%,
discounted at a CAPM cost of equity (10y Treasury + beta × 5% ERP, beta clamped
to [0.5, 2.0], the rate to [7%, 15%]) and compared straight against market cap —
no net-debt bridge. FCFF/WACC would be more orthodox but needs yfinance fields
that come back empty on too many tickers.

It answers three questions per company and adds them to the JSON under
`valuations`, `valuation_method` and `valuation_report`:

1. `implied_growth_pct` — the reverse DCF. Bisection for the FCF growth that
   makes equity value equal today's market cap. **This is what the price
   assumes, not a forecast.**
2. `historical_growth_pct` — the actual FCF CAGR, and the DCF that comes from
   projecting it.
3. `probability` — P(growth ≥ implied) under a Student-t fitted to the company's
   own year-on-year growth.

Things that will bite you:

- **Growth statistics must stay in log space.** Arithmetic year-on-year growth
  does not cancel over a collapse and rebound: Newmont's FCF of
  [1089, 97, 2961, 7299] averages to *+1,003% a year* with a stdev of 1,693%,
  and any probability built on that is noise wearing a lab coat. In logs the
  round trip cancels and the geometric mean comes back equal to the CAGR — which
  is the invariant to test against if you touch this.
- **The R² test is what stops the model inventing trends.** A projection is only
  made when a least-squares line through log FCF clears
  `R2_MINIMO_PARA_PROYECTAR`. On a typical week that is one company out of five.
  That is the correct outcome, not a bug to tune away: with four annual points, a
  poor fit means the "growth rate" describes the path the series took rather than
  where the business is going. CAGR alone cannot see this, because it only looks
  at the endpoints — which is exactly how Newmont produced an 88% CAGR out of a
  trough year.
- **Anything downstream of a failed trend must be withheld, not softened.** The
  earlier version clamped the growth to a band and published the result with a
  footnote; that produced a $8,335 fair value against a $128 price, a number
  driven by the boundary rather than the company. `dcf_value_per_share`,
  `dcf_upside_pct` and `gap_pp` are all null when the trend fails, with
  `dcf_skipped_reason` carrying the explanation. Subtracting from a discarded
  trend gives arithmetic, not evidence.
- **The probability is withheld above `LOG_STDEV_MAX_PUBLICABLE`.** A figure of
  52.3% next to a "treat with caution" label still reads as 52.3% — in a table
  the number always beats the caveat. Below the threshold it is published as a
  range, never a point, because n is 3.
- **n is 3, sometimes 4, and quarterly data does not rescue it.** yfinance
  exposes about five quarters of cash flow, heavily seasonal, and none at all for
  several IBEX tickers. This was checked; do not re-litigate it by reaching for
  `quarterly_cashflow`. The Student-t is doing real work at this sample size — a
  normal would understate the tails badly. `scipy` is deliberately not a
  dependency; the t CDF is a ~40-line incomplete beta.
- **The risk-free rate must match the currency of the cash flows.** Yahoo only
  quotes US Treasuries — there is no Bund and no euro curve — so only USD is
  live and every other currency falls back to a documented constant in
  `TASA_LIBRE_RIESGO_FALLBACK`, surfaced per company in `risk_free_source`.
  Discounting a euro reporter at the US 10-year, which the first version did, is
  not an approximation; it mixes two inflation regimes.
- **Revenue is carried as corroboration, not decoration.** Where FCF growth and
  revenue growth agree the trend is probably real; where they split, the cash
  flow move likely came from working capital, a capex pause or something
  non-recurring. The model cannot tell which — it never sees margins, segments or
  guidance — so the divergence is reported and left uninterpreted.
- **`dcf_terminal_value_share_pct` exists because 2.5% is applied to everything.**
  A miner, a biotech and a beauty retailer do not share a long-run growth
  ceiling. When the terminal share is high, that single assumption is most of the
  answer.
- Negative latest FCF, a currency mismatch between quote and filings, or fewer
  than two years of history all raise and land in `valuation_failed` rather than
  producing a number built on gaps.
- Like the screening report, a failure here never costs the week's data.

## Secrets

`ANTHROPIC_API_KEY` is a **GitHub Actions repository secret on this repo** — not
a Vercel env var. The portal has its own separate copy for its own Claude call.
Two stores, two keys, different execution homes.

```
gh secret list -R jaudi/sp500-quality-screener
```

`GROQ_API_KEY` is left over from before the Claude migration (2026-09-06) and is
read by nothing.
