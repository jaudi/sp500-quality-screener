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
) -> tuple[list, list]:
    """
    Filtra empresas por 6 criterios:
    Fundamentales:
      1. ROE > 20%
      2. ROA > 12%
      3. P/E < 20
      4. Deuda/Patrimonio < 100% (evita "quality traps" apalancados)
    Técnicos:
      5. RSI (14 días) > 30 (excluye sobreventa/distress; sin tope superior
         para no descartar los nombres con momentum más fuerte)
      6. Precio actual > Media móvil de 50 días (confirma tendencia alcista)

    Devuelve (ganadores, fallidos):
      - ganadores: lista de dicts con las empresas que pasaron los 6 filtros
      - fallidos:  lista de dicts {ticker, error} para tickers que no se
                   pudieron evaluar (error de API), distintos de los que
                   simplemente no cumplieron los criterios.
    """
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

            if not (pe and roe and roa and deuda_patrimonio is not None):
                descartados += 1
                continue
            if not ((0 < pe < 20) and (roe > 0.20) and (roa > 0.12) and (deuda_patrimonio < 100)):
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
                    "roa": f"{round(roa * 100, 2)}%",
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
    print(f"   Cumplieron 6 filtros: {len(ganadores)}")
    print(f"   Descartados (no cumplieron criterios): {descartados}")
    print(f"   Fallidos (error de API, no evaluados):  {len(fallidos)}")

    if fallidos:
        print("\n⚠️  Tickers que fallaron y NO se evaluaron (revisar si son falsos negativos):")
        print(", ".join(f["ticker"] for f in fallidos))

    print(f"\n✅ Empresas que superaron los 6 filtros ({len(ganadores)}):\n")
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
def generar_informe(empresas_seleccionadas: list, universo_nombre: str) -> str:
    client = _groq_client()
    prompt_analista = f"""
Eres un analista de inversiones senior. Hemos filtrado {universo_nombre} usando 6 criterios:

Fundamentales:
- ROE > 20%
- ROA > 12%
- P/E < 20
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
# RUNNER GENÉRICO — usado por cada screener.py de índice
# ==============================================================================
def run_pipeline(
    obtener_tickers_fn,
    output_filename: str,
    universo_nombre: str,
    limite_analisis_default: int = 500,
    pausa_entre_tickers: float = 0.4,
) -> str:
    """Ejecuta el cribado completo para un índice y escribe el JSON de salida.

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
    )

    report_text = None
    if empresas_seleccionadas:
        report_text = generar_informe(empresas_seleccionadas, universo_nombre)
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
