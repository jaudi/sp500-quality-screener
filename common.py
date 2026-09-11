"""
Lógica compartida de la pipeline de quality screeners (S&P 500, IBEX 35, ...).

Cada índice tiene su propio script (screener.py, screener_ibex35.py) que solo
define cómo obtener su lista de tickers; todo lo demás — indicadores técnicos,
filtro de calidad, búsqueda web y el agente Claude — vive aquí.
"""

import json
import os
import random
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
import anthropic
from ddgs import DDGS

import valuation
from valuation import analizar_valoraciones

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


MODELO = "claude-sonnet-5"

# Tope de vueltas del bucle agéntico. El trabajo corre desatendido cada semana y
# se factura por token: sin este tope, un modelo que insistiera en buscar podría
# encadenar llamadas indefinidamente.
MAX_ITERACIONES_AGENTE = 25


def _claude_client() -> anthropic.Anthropic:
    """Crea el cliente Claude de forma perezosa (solo cuando hace falta generar el informe)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("⚠️ La variable de entorno ANTHROPIC_API_KEY no está definida.")
    return anthropic.Anthropic(api_key=api_key)


def _texto_de(response) -> str:
    """Concatena los bloques de texto de una respuesta, ignorando thinking y tool_use."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ==============================================================================
# INDICADORES TÉCNICOS (RSI Y MEDIA MÓVIL 50)
# ==============================================================================
def calcular_rsi(precios_cierre: pd.Series, periodo: int = 14) -> float:
    """Calcula el RSI (14 días) a partir de una serie de precios de cierre."""
    delta = precios_cierre.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)

    avg_ganancia = ganancia.rolling(window=periodo).mean()
    avg_perdida = perdida.rolling(window=periodo).mean()

    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)


def calcular_indicadores_tecnicos(ticker: str) -> dict | None:
    """
    Descarga histórico de precios y calcula:
    - RSI (14 días)
    - Precio actual vs. media móvil de 50 días
    Devuelve None si no hay suficiente histórico.
    """
    hist = yf.Ticker(ticker).history(period="4mo", interval="1d")

    if hist.empty or len(hist) < 50:
        return None

    cierre = hist["Close"]
    rsi = calcular_rsi(cierre)
    ma50 = cierre.rolling(window=50).mean().iloc[-1]
    precio_actual = cierre.iloc[-1]
    precio_sobre_ma50 = precio_actual > ma50

    return {
        "rsi": rsi,
        "precio_actual": round(precio_actual, 2),
        "ma50": round(ma50, 2),
        "sobre_ma50": bool(precio_sobre_ma50),
    }


# ==============================================================================
# FILTRO DE CALIDAD — FUNDAMENTAL + TÉCNICO (YFINANCE)
# ==============================================================================
def obtener_info_con_reintentos(ticker: str, max_reintentos: int = 3):
    """
    Envuelve ticker_obj.info con reintentos + backoff exponencial.
    Lanza la última excepción si todos los intentos fallan.
    """
    ultimo_error = None
    for intento in range(max_reintentos):
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            ultimo_error = e
            espera = (2**intento) + random.uniform(0, 0.5)  # backoff exponencial + jitter
            time.sleep(espera)
    raise ultimo_error


def filtrar_acciones_calidad(
    tickers: list,
    limite_analisis: int = 500,
    pausa_entre_tickers: float = 0.4,
    verbose_errores: bool = True,
    roa_minimo: float | None = 0.12,
) -> tuple[list, list]:
    """
    Filtra empresas por criterios fundamentales y técnicos:
    Fundamentales:
      1. ROE > 20%
      2. ROA > roa_minimo (opcional — ver abajo)
      3. P/E < 20
      4. Deuda/Patrimonio < 100% (evita "quality traps" apalancados)
    Técnicos:
      5. RSI (14 días) > 30 (excluye sobreventa/distress; sin tope superior
         para no descartar los nombres con momentum más fuerte)
      6. Precio actual > Media móvil de 50 días (confirma tendencia alcista)

    roa_minimo controla si ROA es un filtro duro o solo informativo:
      - float (ej. 0.12): exige ROA > roa_minimo, igual que el resto de
        fundamentales. Es el comportamiento por defecto (pensado para el
        S&P 500, con perfil growth/tech).
      - None: ROA se sigue calculando e incluyendo en el resultado, pero no
        descarta a nadie. Pensado para índices con más bancos/utilities
        (ej. IBEX 35), donde un ROA bajo es estructural del sector y no una
        señal real de mala calidad. Puede salir "N/A" si yfinance no lo reporta.

    Devuelve (ganadores, fallidos):
      - ganadores: lista de dicts con las empresas que pasaron todos los filtros
      - fallidos:  lista de dicts {ticker, error} para tickers que no se
                   pudieron evaluar (error de API), distintos de los que
                   simplemente no cumplieron los criterios.
    """
    num_filtros = 6 if roa_minimo is not None else 5
    muestra = tickers[:limite_analisis]
    print(f"\n🔍 Analizando fundamentales y técnicos de los primeros {len(muestra)} tickers...")

    ganadores = []
    fallidos = []
    descartados = 0

    for i, t in enumerate(muestra, start=1):
        try:
            info = obtener_info_con_reintentos(t)

            pe = info.get("trailingPE") or info.get("forwardPE")
            roe = info.get("returnOnEquity")
            roa = info.get("returnOnAssets")
            deuda_patrimonio = info.get("debtToEquity")

            campos_requeridos = (pe, roe, deuda_patrimonio is not None)
            if roa_minimo is not None:
                campos_requeridos = (*campos_requeridos, roa)
            if not all(campos_requeridos):
                descartados += 1
                continue
            if not ((0 < pe < 20) and (roe > 0.20) and (deuda_patrimonio < 100)):
                descartados += 1
                continue
            if roa_minimo is not None and not (roa > roa_minimo):
                descartados += 1
                continue

            tecnicos = calcular_indicadores_tecnicos(t)
            if tecnicos is None:
                descartados += 1
                continue
            if not (tecnicos["rsi"] > 30):
                descartados += 1
                continue
            if not tecnicos["sobre_ma50"]:
                descartados += 1
                continue

            ganadores.append(
                {
                    "ticker": t,
                    "nombre": info.get("shortName", t),
                    "sector": info.get("sector", "N/A"),
                    "per": round(pe, 2),
                    "roe": f"{round(roe * 100, 2)}%",
                    "roa": f"{round(roa * 100, 2)}%" if roa is not None else "N/A",
                    "deuda_patrimonio": f"{round(deuda_patrimonio, 1)}%",
                    "rsi": tecnicos["rsi"],
                    "precio_actual": tecnicos["precio_actual"],
                    "ma50": tecnicos["ma50"],
                }
            )

        except Exception as e:
            fallidos.append({"ticker": t, "error": f"{type(e).__name__}: {e}"})
            if verbose_errores:
                print(f"⚠️  [{i}/{len(muestra)}] {t}: fallo tras reintentos — {type(e).__name__}: {e}")
            continue
        finally:
            time.sleep(pausa_entre_tickers)

        if i % 25 == 0:
            print(f"   ...progreso: {i}/{len(muestra)} tickers procesados")

    df_resumen = pd.DataFrame(ganadores)
    print("\n📊 Resumen del cribado:")
    print(f"   Total analizado:      {len(muestra)}")
    print(f"   Cumplieron {num_filtros} filtros: {len(ganadores)}")
    print(f"   Descartados (no cumplieron criterios): {descartados}")
    print(f"   Fallidos (error de API, no evaluados):  {len(fallidos)}")

    if fallidos:
        print("\n⚠️  Tickers que fallaron y NO se evaluaron (revisar si son falsos negativos):")
        print(", ".join(f["ticker"] for f in fallidos))

    print(f"\n✅ Empresas que superaron los {num_filtros} filtros ({len(ganadores)}):\n")
    if not df_resumen.empty:
        print(df_resumen.to_string(index=False))
    else:
        print("Ninguna empresa cumplió los criterios en la muestra analizada.")

    return ganadores, fallidos


# ==============================================================================
# HERRAMIENTAS Y ESQUEMA TOOL-USE (FORMATO ANTHROPIC)
# ==============================================================================
def buscar_noticias_web(ticker: str) -> str:
    """Herramienta de búsqueda web para recopilar noticias del ticker."""
    query = f"{ticker} stock financial news recent catalyst risks performance"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            resumen = [{"titulo": r.get("title"), "fragmento": r.get("body")} for r in results]
            return json.dumps(resumen)
    except Exception as e:
        return json.dumps({"error": f"Error buscando {ticker}: {str(e)}"})


tools = [
    {
        "name": "buscar_noticias_web",
        "description": "Busca noticias recientes, catalizadores financieros y riesgos en la web para un ticker bursátil específico.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "El símbolo bursátil a investigar (ej. 'JNJ', 'AAPL', 'SAN.MC').",
                }
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


# ==============================================================================
# BUCLE AGÉNTICO CON CLAUDE
# ==============================================================================
def _mensaje(client, messages: list, usar_tools: bool = True):
    """Una vuelta del bucle agéntico.

    El SDK de Anthropic ya reintenta por su cuenta los 429 y los 5xx con backoff
    exponencial, así que aquí no hace falta el reintento manual que necesitaba
    Groq (gpt-oss-120b colaba tokens del formato harmony dentro del nombre de la
    función y provocaba un 400 tool_use_failed; ese fallo no existe en Claude).
    """
    extra = {"tools": tools} if usar_tools else {}

    return client.messages.create(
        model=MODELO,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=messages,
        **extra,
    )


def generar_informe(empresas_seleccionadas: list, universo_nombre: str, roa_minimo: float | None = 0.12) -> str:
    client = _claude_client()
    num_criterios = 6 if roa_minimo is not None else 5
    linea_roa = f"- ROA > {round(roa_minimo * 100)}%\n" if roa_minimo is not None else ""
    prompt_analista = f"""
You are a senior investment analyst. We have screened {universo_nombre} using {num_criterios} criteria:

Fundamentals:
- ROE > 20%
{linea_roa}- P/E < 20
- Debt/Equity < 100% (screens out companies whose ROE is inflated by excessive leverage)

Technicals:
- RSI (14-day) > 30 (excludes oversold/distressed names; no upper bound, so strong momentum is not penalised)
- Current price above the 50-day moving average (confirms an uptrend)

Selected companies:
{json.dumps(empresas_seleccionadas, indent=2)}

Instructions:
1. Use the search tool to research the current state and recent news for EACH of these companies.
2. Produce a structured executive report containing:
   - A qualitative summary of each company (growth catalysts vs current risks).
   - An assessment of momentum, grounded in the RSI and the position relative to the MA50 already calculated above.
   - A closing conviction ranking with the reasoning behind it.

Rules:
- Write the entire report in English.
- Only state facts you can support from the screening data above or from what the search tool actually returns. Do not invent revenue figures, earnings numbers, partnerships, deal values, drug names or clinical trial details. If the searches return little, say so and keep the analysis to the screened metrics.
- This is research commentary, not investment advice. Do not recommend buying, selling or holding, and do not suggest position sizes or portfolio weightings.
"""
    messages = [{"role": "user", "content": prompt_analista}]

    print("\n" + "=" * 70)
    print(f"🤖 Agente Claude ({MODELO}) activado: Analizando en tiempo real...")
    print("=" * 70 + "\n")

    for _ in range(MAX_ITERACIONES_AGENTE):
        response = _mensaje(client, messages)

        if response.stop_reason == "refusal":
            raise RuntimeError("El modelo declinó generar el informe de inversión.")

        bloques_tool = [b for b in response.content if b.type == "tool_use"]

        if not bloques_tool:
            informe = _texto_de(response)
            print("\n" + "=" * 70)
            print("📊 INFORME FINAL DE INVERSIÓN GENERADO (CLAUDE):")
            print("=" * 70 + "\n")
            print(informe)
            return informe

        # Se reenvía response.content entero, no solo el texto: los bloques de
        # thinking tienen que volver intactos para que el modelo mantenga su
        # razonamiento entre vueltas.
        messages.append({"role": "assistant", "content": response.content})

        resultados = []
        for bloque in bloques_tool:
            if bloque.name == "buscar_noticias_web":
                ticker_busqueda = bloque.input.get("ticker")
                print(f"🔍 [Web Search] Claude está buscando noticias de: {ticker_busqueda}...")
                contenido = buscar_noticias_web(ticker_busqueda)
                es_error = False
            else:
                # Toda tool call necesita respuesta: si no la añadimos, la
                # siguiente llamada falla por un tool_use_id huérfano.
                print(f"⚠️  Herramienta desconocida solicitada por el modelo: {bloque.name!r}")
                contenido = json.dumps({"error": f"herramienta desconocida: {bloque.name}"})
                es_error = True

            resultados.append({
                "type": "tool_result",
                "tool_use_id": bloque.id,
                "content": contenido,
                "is_error": es_error,
            })

        # Todos los tool_result van en un único mensaje de usuario.
        messages.append({"role": "user", "content": resultados})

    raise RuntimeError(
        f"El agente superó {MAX_ITERACIONES_AGENTE} vueltas sin cerrar el informe."
    )


# ==============================================================================
# AGENTE DE VALORACIÓN — DCF INVERSO VS DCF HISTÓRICO
# ==============================================================================
def generar_informe_valoracion(valoraciones: list, universo_nombre: str) -> str:
    """Comentario de segundo nivel sobre la brecha entre lo que el precio asume
    y lo que la empresa ha hecho históricamente.

    No usa la herramienta de búsqueda: todo lo que necesita ya está calculado en
    valuation.py, y dejar que buscara noticias solo invitaría a mezclar narrativa
    con la aritmética, que es justo lo que este informe intenta separar.
    """
    client = _claude_client()
    prompt_analista = f"""
You are a valuation analyst writing in the spirit of Howard Marks's second-level
thinking: the question is not "is this a good company?" but "what does today's
price already assume, and how often has that actually happened?"

For each company that passed the {universo_nombre} quality screen, a two-stage
DCF has been run on levered free cash flow:

- Discount rate: CAPM cost of equity (10-year Treasury + beta x 5% equity risk
  premium), with beta clamped to [0.5, 2.0] and the resulting rate to [7%, 15%].
- Horizon: 10 explicit years plus a Gordon terminal value at 2.5% perpetual growth.
- `implied_growth_pct` is the REVERSE DCF: the annual FCF growth rate that makes
  the model's equity value equal today's market capitalisation. It is what the
  market is pricing in, not a forecast.
- `historical_growth_pct` is the company's actual FCF CAGR over the years available.
- `modelled_growth_pct` is what the forward DCF actually projected. When
  `historical_growth_capped` is true it was clamped to the [-15%, +25%] band,
  because an unclamped cyclical CAGR produces a fantasy valuation.
- `dcf_value_per_share` / `dcf_upside_pct` come from projecting `modelled_growth_pct`.
- `probability.probability_pct` is P(growth >= implied growth) under a Student-t
  fitted to that company's own year-on-year FCF growth in LOG space, with
  `probability.observations` data points. Treat it as a rough base rate from a
  very small sample, never as a market-implied probability.
- `probability.historical_log_stdev` is the volatility of that growth. It is the
  single best guide to how much the probability is worth: below ~0.2 the company
  compounds steadily and the number means something; above ~1.0 the cash flows
  swing so violently that the probability is barely better than a coin flip, and
  you should say so rather than quoting it as though it were precise.

Valuation data:
{json.dumps(valoraciones, indent=2, ensure_ascii=False, default=str)}

Instructions:
1. For each company, state plainly what the price is assuming, how that compares
   with what the business has actually delivered, and which way the gap cuts.
2. Interpret the probability honestly. A high number means this company's own
   history cleared that bar often; it says nothing about whether the future will.
3. Rank the companies by how undemanding their embedded expectations are — the
   widest favourable gap between what is priced in and what history delivered.
4. Close with a section on where this model is most likely to be wrong.

Rules:
- Write the entire report in English.
- Every company whose `historical_growth_capped` is true must be flagged as such
  in its own section, with the reason the raw CAGR was not projectable.
- Only use the numbers above. Do not invent revenue, margins, guidance, segment
  detail or news. You have no search tool here and no other source.
- A negative `implied_growth_pct` means the price is assuming the business
  shrinks — say so explicitly rather than calling the stock "cheap".
- Flag small-sample fragility wherever `probability.observations` is under 4.
- This is research commentary, not investment advice. Do not recommend buying,
  selling or holding, and do not suggest position sizes or portfolio weightings.
"""
    messages = [{"role": "user", "content": prompt_analista}]

    print("\n" + "=" * 70)
    print(f"🤖 Agente Claude ({MODELO}) activado: analizando la brecha de valoración...")
    print("=" * 70 + "\n")

    response = _mensaje(client, messages, usar_tools=False)

    if response.stop_reason == "refusal":
        raise RuntimeError("El modelo declinó generar el informe de valoración.")

    contenido = _texto_de(response)
    print(contenido)
    return contenido


# ==============================================================================
# SCREENER DE FONDOS TRANSPARENTES (ETFs UCITS) — API PÚBLICA DE ISHARES
# ==============================================================================
ISHARES_PRODUCT_DATA_URL = (
    "https://www.ishares.com/varnish-api/blk-product-screener-server/api/v1/"
    "product-screener/product-data?country=gb&language=en&siteName=ishares-uk&userType=individual"
)


def obtener_universo_ishares() -> dict:
    """Descarga el catálogo público completo de productos iShares (BlackRock).

    Es la misma API JSON que alimenta su propio buscador de fondos para
    inversores particulares (ishares.com/uk/individual/en/products/product-list) —
    no es un scrape de un tercero, es la fuente oficial del fabricante.
    """
    resp = requests.get(ISHARES_PRODUCT_DATA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _resolver_ticker_preferir_lse(nombre_fondo: str, isin: str) -> tuple[str | None, str | None]:
    """Resuelve un fondo a un ticker de Yahoo Finance, prefiriendo su cotización en LSE.

    Un mismo ISIN puede cotizar en varias bolsas (Londres, Ámsterdam, Fráncfort, Milán...).
    yf.Ticker(isin) devuelve una cotización arbitraria, no necesariamente la de Londres —
    la relevante para un inversor que opera desde una plataforma británica (ej. Hargreaves
    Lansdown). Buscar por NOMBRE (no por ISIN) sí expone las distintas cotizaciones, así que
    filtramos esa lista por exchange == "LSE" antes de caer a la resolución directa por ISIN.
    """
    try:
        resultados = yf.Search(nombre_fondo, max_results=10).quotes
        lse = [r for r in resultados if r.get("exchange") == "LSE"]
        if lse:
            return lse[0]["symbol"], "LSE"
    except Exception:
        pass
    try:
        info = yf.Ticker(isin).info
        return info.get("symbol"), info.get("exchange")
    except Exception:
        return None, None


def filtrar_fondos_transparentes(
    domicilios_validos: tuple[str, ...] = ("Ireland", "United Kingdom", "Luxembourg"),
    ter_max: float = 0.20,
    asset_class: str = "Equity",
    pausa_entre_tickers: float = 0.3,
) -> tuple[list, list]:
    """Filtra el catálogo de iShares por vehículo, domicilio y comisión, y calcula
    el Sharpe ratio de cada ETF resultante a partir de su histórico de precios.

    Filtros de transparencia:
      1. Vehículo: solo ETFs (productType == ISHARES_FUND_DATA) — se excluyen los
         fondos indexados tradicionales (BLK_MUTUAL_FUND_DATA) y los ETPs/ETCs.
      2. Domicilio: solo domicilios_validos. Nota: iShares no domicilia ningún
         producto en España — solo Irlanda, Reino Unido, Luxemburgo, Alemania y
         Suiza existen como opciones reales en su catálogo.
      3. Comisión (TER/OCF) < ter_max, tomado directamente del campo oficial de
         iShares — si no está disponible, el fondo se descarta (no se estima).
      4. Clase de activo == asset_class (por defecto Equity, para que el ranking
         por Sharpe compare fondos con perfiles de riesgo comparables).

    Cálculo del Sharpe ratio: rendimiento anualizado / volatilidad anualizada,
    ambos calculados sobre 3 años de precios diarios de cierre, con tipo libre
    de riesgo = 0% (simplificación explícita, no una tasa real del mercado).

    Incluye un filtro de sanidad (-80% a +150% de rendimiento anual plausible)
    para descartar históricos de precio corruptos o splits mal ajustados, y
    deduplica por ticker resuelto (dos ISINs con nombres muy similares pueden
    resolver a la misma cotización por búsqueda de texto).

    Devuelve (ganadores, fallidos), igual que filtrar_acciones_calidad.
    """
    catalogo = obtener_universo_ishares()

    candidatos = []
    for rec in catalogo.values():
        if rec.get("productType") != "ISHARES_FUND_DATA":
            continue
        if rec.get("domicile") not in domicilios_validos:
            continue
        if rec.get("aladdinAssetClass") != asset_class:
            continue
        ter_ocf = rec.get("ter_ocf")
        ter_val = ter_ocf.get("r") if isinstance(ter_ocf, dict) else None
        if ter_val is None or ter_val >= ter_max:
            continue
        ticker_local = rec.get("localExchangeTicker")
        if not ticker_local or ticker_local == "-":
            continue
        candidatos.append(
            {
                "isin": rec.get("isin"),
                "name": rec.get("fundName"),
                "domicile": rec.get("domicile"),
                "ter": ter_val,
            }
        )

    print(f"\n🔍 {len(candidatos)} ETFs candidatos tras filtrar por vehículo, domicilio y TER...")

    ganadores = []
    fallidos = []
    vistos = set()

    for i, c in enumerate(candidatos, start=1):
        try:
            symbol, listado = _resolver_ticker_preferir_lse(c["name"], c["isin"])
            if not symbol or symbol in vistos:
                fallidos.append({"isin": c["isin"], "error": "ticker no resuelto o duplicado"})
                continue

            info = yf.Ticker(symbol).info
            if info.get("quoteType") != "ETF":
                fallidos.append({"isin": c["isin"], "error": f"quoteType={info.get('quoteType')}"})
                continue

            hist = yf.Ticker(symbol).history(period="3y", interval="1d")
            if len(hist) < 500:
                fallidos.append({"isin": c["isin"], "error": f"histórico insuficiente ({len(hist)} filas)"})
                continue

            rets = hist["Close"].pct_change(fill_method=None).dropna()
            rendimiento_anual = (1 + rets.mean()) ** 252 - 1
            volatilidad_anual = rets.std() * (252**0.5)

            if not (-0.80 <= rendimiento_anual <= 1.50) or volatilidad_anual <= 0:
                fallidos.append({"isin": c["isin"], "error": "rendimiento fuera de rango plausible (dato sospechoso)"})
                continue

            vistos.add(symbol)
            ganadores.append(
                {
                    "isin": c["isin"],
                    "ticker": symbol,
                    "listado_lse": listado == "LSE",
                    "nombre": info.get("longName") or c["name"],
                    "domicilio": c["domicile"],
                    "ter": c["ter"],
                    "rendimiento_3y": round(rendimiento_anual * 100, 2),
                    "volatilidad_3y": round(volatilidad_anual * 100, 2),
                    "sharpe": round(rendimiento_anual / volatilidad_anual, 3),
                }
            )
        except Exception as e:
            fallidos.append({"isin": c["isin"], "error": f"{type(e).__name__}: {e}"})
        finally:
            time.sleep(pausa_entre_tickers)

        if i % 25 == 0:
            print(f"   ...progreso: {i}/{len(candidatos)} fondos procesados, {len(ganadores)} válidos")

    ganadores.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n✅ {len(ganadores)} ETFs válidos con Sharpe calculado ({len(fallidos)} descartados/fallidos)")
    return ganadores, fallidos


def generar_informe_fondos(fondos_top: list, universo_nombre: str, ter_max: float = 0.20) -> str:
    """Genera un comentario cualitativo con Claude sobre el top de ETFs por Sharpe ratio.

    A diferencia de generar_informe (acciones), no usa la herramienta de búsqueda web —
    son ETFs indexados pasivos, no hay "noticias" por fondo que investigar; el análisis
    se apoya en los propios datos de rendimiento/volatilidad/TER ya calculados.
    """
    client = _claude_client()
    prompt_analista = f"""
You are a senior investment analyst specialising in European UCITS ETFs.

We have screened the public iShares (BlackRock) fund catalogue against these
transparency criteria:
- Vehicle: ETFs only (no traditional index funds, no ETPs/ETCs)
- Domicile: Ireland, United Kingdom or Luxembourg (recognised UCITS jurisdictions)
- Fee (TER/OCF): below {ter_max * 100:.0f}%
- Asset class: Equity
- Preferred listing: London Stock Exchange (LSE) where one exists, for
  accessibility to investors trading from UK/European platforms

The ranking uses the Sharpe ratio (annualised return / annualised volatility,
risk-free rate = 0%, computed over 3 years of daily prices) as the measure of
risk-adjusted performance.

The {len(fondos_top)} funds with the best 3-year Sharpe ratio:
{json.dumps(fondos_top, indent=2, ensure_ascii=False)}

Instructions:
1. For each fund, briefly comment on what its index/exposure represents and why
   its return/volatility combination produced this Sharpe ratio.
2. Flag any thematic concentration or notable bias across the set (for example
   overexposure to one sector, region or currency).
3. Close with a conclusion on what kind of investor might find most value in this
   ranking, stating explicitly that it is a historical 3-year risk/return ranking
   — not a buy recommendation and not a projection of future performance.

Rules:
- Write the entire commentary in English.
- Only state facts supported by the fund data above. Do not invent performance
  figures, holdings, fund sizes or index details you cannot derive from it.
- Do not recommend buying, selling or holding, and do not suggest position sizes.
"""
    messages = [{"role": "user", "content": prompt_analista}]

    print("\n" + "=" * 70)
    print(f"🤖 Agente Claude ({MODELO}) activado: analizando el ranking de fondos...")
    print("=" * 70 + "\n")

    response = _mensaje(client, messages, usar_tools=False)

    if response.stop_reason == "refusal":
        raise RuntimeError("El modelo declinó generar el comentario de fondos.")

    contenido = _texto_de(response)
    print(contenido)
    return contenido


def run_pipeline_fondos(
    output_filename: str = "latest-report-funds.json",
    universo_nombre: str = "transparent iShares ETFs (IE/GB/LU, TER<0.20%)",
    domicilios_validos: tuple[str, ...] = ("Ireland", "United Kingdom", "Luxembourg"),
    ter_max: float = 0.20,
    top_n: int = 10,
) -> str:
    """Ejecuta el cribado completo de fondos y escribe el JSON de salida.

    Devuelve la ruta del archivo escrito.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("⚠️ La variable de entorno ANTHROPIC_API_KEY no está definida.")

    ganadores, fallidos = filtrar_fondos_transparentes(
        domicilios_validos=domicilios_validos,
        ter_max=ter_max,
    )
    top_fondos = ganadores[:top_n]

    report_text = None
    if top_fondos:
        report_text = generar_informe_fondos(top_fondos, universo_nombre, ter_max)
    else:
        print("Ningún fondo cumplió los filtros esta semana — se omite la llamada a Claude.")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "data_source": "iShares (BlackRock) public product-screener API — the same feed that "
            "powers ishares.com's own retail fund finder",
            "vehicle": "ETF only (excludes traditional index funds and ETPs/ETCs)",
            "domicile": list(domicilios_validos),
            "domicile_note": "iShares has no Spain-domiciled products — only Ireland, UK, "
            "Luxembourg, Germany and Switzerland exist in their catalogue",
            "max_ter_ocf_pct": ter_max,
            "asset_class": "Equity",
            "listing_preference": "London Stock Exchange (LSE) preferred when available, for "
            "buyability on UK/European retail platforms",
            "sharpe_calc": "3-year annualized return ÷ 3-year annualized volatility, from daily "
            "close prices, risk-free rate assumed 0%",
            "sanity_filter": "annualized return outside -80%..+150% is treated as corrupt price "
            "data and discarded, not shown",
        },
        "universe_size": len(ganadores) + len(fallidos),
        "passed_filters": len(ganadores),
        "funds": top_fondos,
        "failed_count": len(fallidos),
        "report": report_text,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Informe escrito en {output_path}")
    return output_path


# ==============================================================================
# RUNNER GENÉRICO — usado por cada screener.py de índice
# ==============================================================================
def run_pipeline(
    obtener_tickers_fn,
    output_filename: str,
    universo_nombre: str,
    limite_analisis_default: int = 500,
    pausa_entre_tickers: float = 0.4,
    roa_minimo: float | None = 0.12,
) -> str:
    """Ejecuta el cribado completo para un índice y escribe el JSON de salida.

    roa_minimo: ver filtrar_acciones_calidad — pásalo como None para índices
    donde ROA no debería ser un filtro duro (ej. IBEX 35).

    Devuelve la ruta del archivo escrito.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("⚠️ La variable de entorno ANTHROPIC_API_KEY no está definida.")

    todos_los_tickers = obtener_tickers_fn()
    limite_analisis = int(os.environ.get("SCREENER_LIMIT", str(limite_analisis_default)))
    empresas_seleccionadas, tickers_fallidos = filtrar_acciones_calidad(
        todos_los_tickers,
        limite_analisis=limite_analisis,
        pausa_entre_tickers=pausa_entre_tickers,
        roa_minimo=roa_minimo,
    )

    report_text = None
    if empresas_seleccionadas:
        try:
            report_text = generar_informe(empresas_seleccionadas, universo_nombre, roa_minimo)
        except Exception as e:
            # El cribado ya tiene valor por sí solo. Si el agente falla, se escribe
            # igualmente el JSON con las empresas seleccionadas y report=None, en vez
            # de perder la ejecución entera y dejar el portal con datos de la semana pasada.
            print(f"⚠️  El agente no pudo generar el informe: {type(e).__name__}: {e}")
            print("   Se guardan igualmente los resultados del cribado, sin informe cualitativo.")
    else:
        print("Ninguna empresa cumplió los filtros esta semana — se omite la llamada a Claude.")

    # Etapa de valoración. Va después del informe cualitativo y en su propio
    # try: son dos preguntas independientes ("¿es buena?" y "¿qué asume el
    # precio?"), así que un fallo aquí no debe costar el informe que ya está
    # escrito, ni al revés.
    valoraciones, valoraciones_fallidas, valuation_report = [], [], None
    if empresas_seleccionadas:
        try:
            valoraciones, valoraciones_fallidas = analizar_valoraciones(
                empresas_seleccionadas,
                pausa_entre_tickers=pausa_entre_tickers,
            )
            if valoraciones:
                valuation_report = generar_informe_valoracion(valoraciones, universo_nombre)
        except Exception as e:
            print(f"⚠️  La etapa de valoración falló: {type(e).__name__}: {e}")
            print("   Se guarda el JSON sin el bloque de valoración.")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(todos_los_tickers),
        "analyzed": min(limite_analisis, len(todos_los_tickers)),
        "passed_filters": len(empresas_seleccionadas),
        "companies": empresas_seleccionadas,
        "failed": tickers_fallidos,
        "report": report_text,
        "valuation_method": {
            "model": "Two-stage DCF on levered free cash flow (FCF discounted at cost of equity, compared with market cap — no net-debt bridge)",
            "horizon_years": valuation.HORIZONTE_ANIOS,
            "terminal_growth_pct": valuation.CRECIMIENTO_TERMINAL * 100,
            "discount_rate": f"CAPM: 10y Treasury + beta x {valuation.PRIMA_RIESGO_MERCADO * 100:.0f}% ERP, beta clamped to [{valuation.BETA_MIN}, {valuation.BETA_MAX}], rate clamped to [{valuation.KE_MIN * 100:.0f}%, {valuation.KE_MAX * 100:.0f}%]",
            "implied_growth": "Reverse DCF — the FCF growth rate that sets the model's equity value equal to today's market cap. What the price assumes, not a forecast.",
            "projected_growth_band_pct": [valuation.G_MODELADO_MIN * 100, valuation.G_MODELADO_MAX * 100],
            "probability": "P(growth >= implied growth) under a Student-t fitted to the company's own year-on-year FCF growth. Small-sample base rate from its own history, not a market-implied probability.",
        },
        "valuations": valoraciones,
        "valuation_failed": valoraciones_fallidas,
        "valuation_report": valuation_report,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Informe escrito en {output_path}")
    return output_path
