"""
Lógica compartida de la pipeline de quality screeners (S&P 500, IBEX 35, ...).

Cada índice tiene su propio script (screener.py, screener_ibex35.py) que solo
define cómo obtener su lista de tickers; todo lo demás — indicadores técnicos,
filtro de calidad, búsqueda web y el agente Groq — vive aquí.
"""

import json
import os
import random
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from ddgs import DDGS
from groq import Groq

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _groq_client() -> Groq:
    """Crea el cliente Groq de forma perezosa (solo cuando hace falta generar el informe)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ La variable de entorno GROQ_API_KEY no está definida.")
    return Groq(api_key=api_key)


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
# HERRAMIENTAS Y ESQUEMA TOOL-USE (FORMATO OPENAI/GROQ)
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
        "type": "function",
        "function": {
            "name": "buscar_noticias_web",
            "description": "Busca noticias recientes, catalizadores financieros y riesgos en la web para un ticker bursátil específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "El símbolo bursátil a investigar (ej. 'JNJ', 'AAPL', 'SAN.MC').",
                    }
                },
                "required": ["ticker"],
            },
        },
    }
]


# ==============================================================================
# BUCLE AGÉNTICO CON GROQ (GPT-OSS-120B)
# ==============================================================================
def generar_informe(empresas_seleccionadas: list, universo_nombre: str, roa_minimo: float | None = 0.12) -> str:
    client = _groq_client()
    num_criterios = 6 if roa_minimo is not None else 5
    linea_roa = f"- ROA > {round(roa_minimo * 100)}%\n" if roa_minimo is not None else ""
    prompt_analista = f"""
Eres un analista de inversiones senior. Hemos filtrado {universo_nombre} usando {num_criterios} criterios:

Fundamentales:
- ROE > 20%
{linea_roa}- P/E < 20
- Deuda/Patrimonio < 100% (evita empresas con ROE inflado por apalancamiento excesivo)

Técnicos:
- RSI (14 días) > 30 (excluye sobreventa/distress, sin tope superior para no penalizar el momentum fuerte)
- Precio actual por encima de la media móvil de 50 días (confirmación de tendencia alcista)

Empresas seleccionadas:
{json.dumps(empresas_seleccionadas, indent=2)}

Instrucciones para el análisis:
1. Usa la herramienta de búsqueda para investigar el estado actual y noticias de CADA una de estas empresas.
2. Genera un informe ejecutivo estructurado que contenga:
   - Resumen cualitativo de cada empresa (catalizadores de crecimiento vs riesgos actuales).
   - Evaluación del momentum, apoyándote en el RSI y la posición respecto a la MA50 ya calculados.
   - Conclusión final con un ranking de convicción fundamentado.
"""

    messages = [{"role": "user", "content": prompt_analista}]

    print("\n" + "=" * 70)
    print("🤖 Agente Groq (gpt-oss-120b) activado: Analizando en tiempo real...")
    print("=" * 70 + "\n")

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=3000,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            print("\n" + "=" * 70)
            print("📊 INFORME FINAL DE INVERSIÓN GENERADO (GROQ):")
            print("=" * 70 + "\n")
            print(response_message.content)
            return response_message.content

        messages.append(response_message)

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id

            if func_name == "buscar_noticias_web":
                ticker_busqueda = func_args.get("ticker")
                print(f"🔍 [Web Search] Groq está buscando noticias de: {ticker_busqueda}...")

                info_noticias = buscar_noticias_web(ticker_busqueda)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": info_noticias,
                    }
                )


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
    """Genera un comentario cualitativo con Groq sobre el top de ETFs por Sharpe ratio.

    A diferencia de generar_informe (acciones), no usa la herramienta de búsqueda web —
    son ETFs indexados pasivos, no hay "noticias" por fondo que investigar; el análisis
    se apoya en los propios datos de rendimiento/volatilidad/TER ya calculados.
    """
    client = _groq_client()
    prompt_analista = f"""
Eres un analista de inversiones senior especializado en ETFs UCITS europeos.

Hemos filtrado el catálogo público de fondos iShares (BlackRock) aplicando estos
criterios de transparencia:
- Vehículo: solo ETFs (no fondos indexados tradicionales ni ETPs/ETCs)
- Domicilio: Irlanda, Reino Unido o Luxemburgo (jurisdicciones UCITS reconocidas)
- Comisión (TER/OCF): inferior al {ter_max * 100:.0f}%
- Clase de activo: Renta Variable (Equity)
- Cotización preferida: Bolsa de Londres (LSE) cuando existe, por accesibilidad
  para inversores que operan desde plataformas británicas/europeas

El ranking usa el Sharpe ratio (rendimiento anualizado ÷ volatilidad anualizada,
tipo libre de riesgo = 0%, calculado sobre 3 años de precios diarios) como medida
de rentabilidad ajustada al riesgo.

Los {len(fondos_top)} fondos con mejor Sharpe ratio de los últimos 3 años:
{json.dumps(fondos_top, indent=2, ensure_ascii=False)}

Instrucciones para el análisis:
1. Para cada fondo, comenta brevemente qué representa su índice/exposición y por
   qué su combinación de rendimiento/volatilidad ha producido este Sharpe ratio.
2. Señala cualquier concentración temática o sesgo relevante en el conjunto (ej.
   sobreexposición a un sector, región o divisa).
3. Cierra con una conclusión sobre qué tipo de inversor podría encontrar más valor
   en este ranking, dejando explícito que es un ranking histórico de riesgo/
   rentabilidad de los últimos 3 años — no una recomendación de compra ni una
   proyección de rendimiento futuro.
"""
    messages = [{"role": "user", "content": prompt_analista}]

    print("\n" + "=" * 70)
    print("🤖 Agente Groq (gpt-oss-120b) activado: analizando el ranking de fondos...")
    print("=" * 70 + "\n")

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        max_tokens=2000,
    )
    contenido = response.choices[0].message.content
    print(contenido)
    return contenido


def run_pipeline_fondos(
    output_filename: str = "latest-report-funds.json",
    universo_nombre: str = "ETFs iShares transparentes (IE/GB/LU, TER<0.20%)",
    domicilios_validos: tuple[str, ...] = ("Ireland", "United Kingdom", "Luxembourg"),
    ter_max: float = 0.20,
    top_n: int = 10,
) -> str:
    """Ejecuta el cribado completo de fondos y escribe el JSON de salida.

    Devuelve la ruta del archivo escrito.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("⚠️ La variable de entorno GROQ_API_KEY no está definida.")

    ganadores, fallidos = filtrar_fondos_transparentes(
        domicilios_validos=domicilios_validos,
        ter_max=ter_max,
    )
    top_fondos = ganadores[:top_n]

    report_text = None
    if top_fondos:
        report_text = generar_informe_fondos(top_fondos, universo_nombre, ter_max)
    else:
        print("Ningún fondo cumplió los filtros esta semana — se omite la llamada a Groq.")

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
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("⚠️ La variable de entorno GROQ_API_KEY no está definida.")

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
        report_text = generar_informe(empresas_seleccionadas, universo_nombre, roa_minimo)
    else:
        print("Ninguna empresa cumplió los filtros esta semana — se omite la llamada a Groq.")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(todos_los_tickers),
        "analyzed": min(limite_analisis, len(todos_los_tickers)),
        "passed_filters": len(empresas_seleccionadas),
        "companies": empresas_seleccionadas,
        "failed": tickers_fallidos,
        "report": report_text,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Informe escrito en {output_path}")
    return output_path
