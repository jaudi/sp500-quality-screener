"""
Lógica compartida de la pipeline de screeners (S&P 500, IBEX 35, Nasdaq-100, ...).

Cada índice tiene su propio script (screener.py, screener_ibex35.py,
screener_nasdaq100.py) que solo define cómo obtener su lista de tickers y con
qué criterios cribarla; todo lo demás — indicadores técnicos, filtros, búsqueda
web y los agentes Claude — vive aquí.

Hay dos cribados, no uno:

- **Calidad** (`filtrar_acciones_calidad`): ROE, P/E, deuda. Pensado para
  índices amplios donde el múltiplo todavía discrimina.
- **Crecimiento** (`filtrar_acciones_crecimiento`): crecimiento de ingresos y
  beneficios, caja libre positiva, y una estructura de tendencia completa.
  Pensado para el Nasdaq-100, donde un filtro de P/E < 20 no dejaría pasar casi
  nada y lo poco que pasara sería precisamente lo que el índice tiene de menos
  característico.
"""

import json
import os
import random
import time
from datetime import datetime, timezone

import pandas as pd
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
# INDICADORES TÉCNICOS (RSI, MEDIAS MÓVILES, MOMENTUM)
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


# Sesiones bursátiles, no días naturales: ~21 al mes. Se usan para mirar atrás
# dentro de la serie, así que tienen que contarse en la misma unidad que la serie.
SESIONES_6M = 126
SESIONES_12M = 252


def calcular_indicadores_tecnicos(ticker: str, con_tendencia_larga: bool = False) -> dict | None:
    """
    Descarga histórico de precios y calcula:
    - RSI (14 días)
    - Precio actual vs. media móvil de 50 días

    Con `con_tendencia_larga=True` añade la estructura de tendencia completa que
    necesita el cribado de momentum: MA200, el cruce MA50/MA200 y los retornos a
    6 y 12 meses. Eso obliga a descargar 14 meses en vez de 4 — más de 200
    sesiones para la media, y margen para el retorno a 12 meses — así que solo se
    pide cuando hace falta: el cribado de calidad solo mira la MA50 y no debería
    pagar la descarga larga en 500 tickers.

    Devuelve None si no hay suficiente histórico para lo que se ha pedido.
    """
    periodo = "14mo" if con_tendencia_larga else "4mo"
    minimo_sesiones = SESIONES_12M if con_tendencia_larga else 50

    hist = yf.Ticker(ticker).history(period=periodo, interval="1d")

    if hist.empty or len(hist) < minimo_sesiones:
        return None

    cierre = hist["Close"]
    rsi = calcular_rsi(cierre)
    ma50 = cierre.rolling(window=50).mean().iloc[-1]
    precio_actual = cierre.iloc[-1]
    precio_sobre_ma50 = precio_actual > ma50

    indicadores = {
        "rsi": rsi,
        "precio_actual": round(precio_actual, 2),
        "ma50": round(ma50, 2),
        "sobre_ma50": bool(precio_sobre_ma50),
    }

    if not con_tendencia_larga:
        return indicadores

    ma200 = cierre.rolling(window=200).mean().iloc[-1]
    if pd.isna(ma200):
        return None

    # El retorno se toma sobre precios ajustados por splits y dividendos, que es
    # lo que devuelve yf.history por defecto: si no, un split parece un -50%.
    retorno_6m = precio_actual / cierre.iloc[-SESIONES_6M] - 1
    retorno_12m = precio_actual / cierre.iloc[-SESIONES_12M] - 1

    indicadores.update(
        {
            "ma200": round(ma200, 2),
            "sobre_ma200": bool(precio_actual > ma200),
            # La MA50 por encima de la MA200 es la tendencia de fondo. Separada
            # de "precio > MA50" a propósito: el precio puede recuperar la MA50
            # en cualquier rebote de dos semanas, el cruce de medias no.
            "ma50_sobre_ma200": bool(ma50 > ma200),
            "retorno_6m": round(retorno_6m * 100, 2),
            "retorno_12m": round(retorno_12m * 100, 2),
        }
    )
    return indicadores


# ==============================================================================
# FILTRO DE CALIDAD — FUNDAMENTAL + TÉCNICO (YFINANCE)
# ==============================================================================
# Yahoo devuelve algunos shortName rotos: truncados a 31 caracteres y con las
# letras no ASCII sustituidas por tres puntos literales. LOG.MC llega como
# "COMPA...IA DE DISTRIBUCION INTE" — 0x2e 0x2e 0x2e donde debería ir la "Ñ".
# El destrozo viene ya hecho dentro de la respuesta de yfinance, así que no es
# un problema de codificación nuestro y no se arregla decodificando distinto.
# longName sí llega limpio, pero no siempre es el nombre que queremos enseñar
# (el de LOG.MC es "Logista Integral, S.A.", que no es la razón social), así
# que los conocidos se fijan a mano aquí. Se aplica después de la consulta a
# yfinance; cualquier ticker que no esté en el mapa conserva lo que devuelva
# Yahoo, así que añadir uno nuevo no toca el resto de la pipeline.
NOMBRES_CORREGIDOS = {
    "ACS.MC": "ACS, Actividades de Construcción y Servicios, S.A.",
    "ANE.MC": "Corporación Acciona Energías Renovables, S.A.",
    "BBVA.MC": "Banco Bilbao Vizcaya Argentaria, S.A.",
    "IAG.MC": "International Consolidated Airlines Group, S.A.",
    "LOG.MC": "Compañía de Distribución Integral Logista Holdings, S.A.",
    "ROVI.MC": "Laboratorios Farmacéuticos Rovi, S.A.",
    "SLR.MC": "Solaria Energía y Medio Ambiente, S.A.",
}


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
                    "nombre": NOMBRES_CORREGIDOS.get(t, info.get("shortName", t)),
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
# FILTRO DE CRECIMIENTO — CRECIMIENTO + MOMENTUM (YFINANCE)
# ==============================================================================
# Umbrales calibrados sobre el Nasdaq-100: los siete juntos dejan pasar ~10-15
# nombres de 100 en un mercado normal, que es el tamaño de informe que se busca.
# No son verdades: son el punto donde el cribado sigue siendo selectivo sin
# quedarse vacío. Si se mueven, muévanse mirando cuántos pasan.
CRECIMIENTO_INGRESOS_MIN = 0.10
CRECIMIENTO_BENEFICIOS_MIN = 0.10
RSI_MINIMO_CRECIMIENTO = 40


def _puntuar_y_ordenar(ganadores: list) -> None:
    """Asigna a cada ganador un `score` 0-100 y ordena la lista in place.

    Es un rango percentil medio sobre tres columnas — crecimiento de ingresos,
    crecimiento de beneficios y retorno a 6 meses — no una puntuación absoluta.
    Un 100 significa "el mejor de los que pasaron el filtro esta semana", no
    "bueno", y el score de una empresa cambia si cambia el resto de la cohorte.

    Se usa el rango y no el valor crudo a propósito: un trimestre con +1.300% de
    crecimiento de beneficios — Micron sale así de un suelo cíclico — dominaría
    cualquier media de valores y convertiría el score en una sola columna
    disfrazada de tres.
    """
    if not ganadores:
        return

    if len(ganadores) == 1:
        ganadores[0]["score"] = 100.0
        return

    df = pd.DataFrame(
        {
            "ingresos": [g["crecimiento_ingresos"] for g in ganadores],
            "beneficios": [g["crecimiento_beneficios"] for g in ganadores],
            "momentum": [g["retorno_6m"] for g in ganadores],
        }
    )
    puntuaciones = df.rank(pct=True).mean(axis=1) * 100
    for ganador, puntuacion in zip(ganadores, puntuaciones):
        ganador["score"] = round(float(puntuacion), 1)

    ganadores.sort(key=lambda g: g["score"], reverse=True)


def filtrar_acciones_crecimiento(
    tickers: list,
    limite_analisis: int = 110,
    pausa_entre_tickers: float = 0.4,
    verbose_errores: bool = True,
    crecimiento_ingresos_min: float = CRECIMIENTO_INGRESOS_MIN,
    crecimiento_beneficios_min: float = CRECIMIENTO_BENEFICIOS_MIN,
    rsi_minimo: float = RSI_MINIMO_CRECIMIENTO,
) -> tuple[list, list]:
    """
    Filtra empresas por crecimiento y momentum — no por calidad ni por múltiplo.

    Crecimiento:
      1. Crecimiento de ingresos interanual > crecimiento_ingresos_min
      2. Crecimiento de beneficios interanual > crecimiento_beneficios_min
      3. Flujo de caja libre positivo

    Momentum:
      4. Precio actual > media móvil de 50 días
      5. MA50 > MA200 (la tendencia de fondo, no un rebote)
      6. Retorno a 6 meses > 0
      7. RSI (14 días) > rsi_minimo (descarta debilidad; sin tope superior, para
         no penalizar precisamente lo que se está buscando)

    Por qué aquí no hay P/E, ROE ni deuda: en el Nasdaq-100 un filtro de P/E < 20
    descarta casi todo el índice, y lo poco que deja pasar es justo lo menos
    representativo de lo que el índice es. Un múltiplo bajo no señala calidad en
    un universo de crecimiento; señala que el mercado ya no espera crecimiento.
    La pregunta que responde este cribado es otra: quién está creciendo, y con el
    precio acompañando.

    El filtro 3 (caja libre positiva) es el único guardarraíl de calidad, y está
    puesto a conciencia: separa el crecimiento que genera caja del que la quema,
    y es además lo que necesita la etapa de valoración que corre después — con un
    FCF negativo falla igualmente.

    Devuelve (ganadores, fallidos), con la misma semántica que
    filtrar_acciones_calidad: `fallidos` son tickers que no se pudieron evaluar
    por un error de API, distintos de los que simplemente no cumplieron los
    criterios. Los ganadores vienen ordenados por `score` descendente.
    """
    num_filtros = 7
    muestra = tickers[:limite_analisis]
    print(f"\n🔍 Analizando crecimiento y momentum de los primeros {len(muestra)} tickers...")

    ganadores = []
    fallidos = []
    descartados = 0

    for i, t in enumerate(muestra, start=1):
        try:
            info = obtener_info_con_reintentos(t)

            crecimiento_ingresos = info.get("revenueGrowth")
            # earningsGrowth viene vacío en bastantes nombres; earningsQuarterlyGrowth
            # es el mismo dato por otra vía. El `is None` explícito importa: un
            # crecimiento de exactamente 0.0 es un dato, no un hueco, y un `or` lo
            # trataría como ausente.
            crecimiento_beneficios = info.get("earningsGrowth")
            if crecimiento_beneficios is None:
                crecimiento_beneficios = info.get("earningsQuarterlyGrowth")
            flujo_caja_libre = info.get("freeCashflow")
            margen_bruto = info.get("grossMargins")

            if crecimiento_ingresos is None or crecimiento_beneficios is None or flujo_caja_libre is None:
                descartados += 1
                continue

            if not (
                crecimiento_ingresos > crecimiento_ingresos_min
                and crecimiento_beneficios > crecimiento_beneficios_min
                and flujo_caja_libre > 0
            ):
                descartados += 1
                continue

            tecnicos = calcular_indicadores_tecnicos(t, con_tendencia_larga=True)
            if tecnicos is None:
                descartados += 1
                continue

            if not (
                tecnicos["sobre_ma50"]
                and tecnicos["ma50_sobre_ma200"]
                and tecnicos["retorno_6m"] > 0
                and tecnicos["rsi"] > rsi_minimo
            ):
                descartados += 1
                continue

            ganadores.append(
                {
                    "ticker": t,
                    "nombre": NOMBRES_CORREGIDOS.get(t, info.get("shortName", t)),
                    "sector": info.get("sector", "N/A"),
                    "crecimiento_ingresos": round(crecimiento_ingresos * 100, 1),
                    "crecimiento_beneficios": round(crecimiento_beneficios * 100, 1),
                    "margen_bruto": round(margen_bruto * 100, 1) if margen_bruto is not None else None,
                    # En millones de la divisa de reporte, que en el Nasdaq-100 es
                    # el dólar en la práctica totalidad de los casos.
                    "flujo_caja_libre": round(flujo_caja_libre / 1e6),
                    "rsi": tecnicos["rsi"],
                    "precio_actual": tecnicos["precio_actual"],
                    "ma50": tecnicos["ma50"],
                    "ma200": tecnicos["ma200"],
                    "retorno_6m": tecnicos["retorno_6m"],
                    "retorno_12m": tecnicos["retorno_12m"],
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

    _puntuar_y_ordenar(ganadores)

    print("\n📊 Resumen del cribado de crecimiento:")
    print(f"   Total analizado:      {len(muestra)}")
    print(f"   Cumplieron {num_filtros} filtros: {len(ganadores)}")
    print(f"   Descartados (no cumplieron criterios): {descartados}")
    print(f"   Fallidos (error de API, no evaluados):  {len(fallidos)}")

    if fallidos:
        print("\n⚠️  Tickers que fallaron y NO se evaluaron (revisar si son falsos negativos):")
        print(", ".join(f["ticker"] for f in fallidos))

    print(f"\n✅ Empresas que superaron los {num_filtros} filtros ({len(ganadores)}):\n")
    df_resumen = pd.DataFrame(ganadores)
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


def _bucle_agentico(client, messages: list) -> str:
    """Bucle agéntico con tool-use. Devuelve el texto del informe.

    Compartido por los informes de calidad y de crecimiento: los dos hacen
    preguntas distintas pero corren exactamente la misma mecánica, y esa mecánica
    tiene tres detalles que se rompen fácil si se duplica (ver CLAUDE.md): el tope
    de vueltas, reenviar `response.content` entero para que sobrevivan los bloques
    de thinking, y responder a TODA tool call — también a las desconocidas — o la
    siguiente llamada falla por un tool_use_id huérfano.
    """
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
{json.dumps(empresas_seleccionadas, indent=2, ensure_ascii=False)}

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

    return _bucle_agentico(client, messages)


def generar_informe_crecimiento(empresas_seleccionadas: list, universo_nombre: str) -> str:
    """Informe cualitativo del cribado de crecimiento.

    Mismo bucle agéntico que `generar_informe`, otra pregunta. Aquel busca
    calidad a buen precio; este tiene que separar el crecimiento que se sostiene
    del que es un efecto de comparación — un trimestre contra un suelo cíclico
    produce cifras espectaculares que no dicen nada sobre el año que viene, y es
    exactamente el sesgo al que un cribado de crecimiento está expuesto.
    """
    client = _claude_client()
    prompt_analista = f"""
You are a senior growth-equity analyst. We have screened {universo_nombre} using 7 criteria.

This is a growth and momentum screen, not a quality or value screen. There is deliberately no P/E, ROE or leverage filter: in this index a low multiple does not signal quality, it signals that the market has stopped expecting growth.

Growth:
- Year-on-year revenue growth > 10%
- Year-on-year earnings growth > 10%
- Positive free cash flow (the only quality guardrail — it separates growth that generates cash from growth that burns it)

Momentum:
- Current price above the 50-day moving average
- 50-day moving average above the 200-day (the underlying trend, not a two-week bounce)
- Positive 6-month total return
- RSI (14-day) > 40 (excludes weakness; no upper bound, so strong momentum is not penalised)

Selected companies:
{json.dumps(empresas_seleccionadas, indent=2, ensure_ascii=False)}

Reading the data:
- Growth figures are percentages, year on year, from the most recent reported period.
- `flujo_caja_libre` is trailing free cash flow in millions of the reporting currency.
- `retorno_6m` and `retorno_12m` are total returns in percent, on split- and dividend-adjusted prices.
- `score` is a within-cohort percentile rank averaged over revenue growth, earnings growth and 6-month return. 100 means "best of the names that passed this week", NOT "best company" and NOT a quality score. Treat it as an ordering of this week's cohort and nothing more.

Instructions:
1. Use the search tool to research the current state and recent news for EACH of these companies.
2. Produce a structured executive report containing:
   - For each company: what is actually driving the growth, and whether the reported rate looks durable or is largely a base effect. A triple-digit earnings growth rate coming off a cyclical trough (semiconductor memory is the classic case) is arithmetic about last year, not evidence about next year — say so plainly where you see it.
   - An assessment of momentum for each name, grounded in the RSI, the position relative to the MA50 and MA200, and the 6- and 12-month returns already calculated above. Note where price has run far ahead of the trend as well as where it confirms it.
   - The main risks to the growth continuing — end-market cycle, customer concentration, a single product cycle, competition.
   - A closing ranking by conviction in the *durability* of the growth, with the reasoning. Say where it differs from the `score` ordering and why.

Rules:
- Write the entire report in English.
- Only state facts you can support from the screening data above or from what the search tool actually returns. Do not invent revenue figures, earnings numbers, partnerships, deal values, product roadmaps or customer names. If the searches return little, say so and keep the analysis to the screened metrics.
- Do not describe a company as cheap, expensive or fairly valued. This screen contains no valuation input at all; a separate reverse-DCF stage runs afterwards and that is where price is addressed.
- This is research commentary, not investment advice. Do not recommend buying, selling or holding, and do not suggest position sizes or portfolio weightings.
"""
    messages = [{"role": "user", "content": prompt_analista}]

    print("\n" + "=" * 70)
    print(f"🤖 Agente Claude ({MODELO}) activado: cribado de crecimiento...")
    print("=" * 70 + "\n")

    return _bucle_agentico(client, messages)


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
  It is also the ONE figure available for every company, because it needs only
  current cash flow and market cap — no history, no trend, no projection. Lead
  with it.
- `historical_growth_pct` is the endpoint-to-endpoint FCF CAGR. Report it, but do
  not lean on it: it sees only the first and last year.
- `trend_growth_pct` and `trend_r2` come from a least-squares line through log
  FCF, using every point. **`trend_r2` is the most important diagnostic here.**
  It says whether the cash flows behave like a trend at all. Below 0.5 the series
  is a path, not a direction, and no projection was made.
- `revenue_growth_pct` and `fcf_vs_revenue_divergence_pp` are the corroboration
  check. Where FCF growth and revenue growth agree, the trend is probably real.
  Where they diverge sharply, the FCF move is more likely working capital, a
  capex pause or something non-recurring — say so, but do NOT claim to know
  which, because you have no margin, segment or guidance data.
- `dcf_value_per_share` / `dcf_upside_pct` are often null, and `dcf_skipped_reason`
  says why. A null is a finding, not a gap to apologise for: the model refused to
  project rather than publish a number driven by its own boundary. Never supply a
  value the model declined to produce, and never describe the refusal as missing
  data.
- `dcf_terminal_value_share_pct` is how much of the DCF comes from the 2.5%
  perpetuity rather than the ten explicit years. Where it is high, most of the
  valuation is that one fixed assumption — applied identically to a miner, a
  biotech and a retailer, which have nothing like the same long-run ceiling.
- `probability.probability_pct` is P(growth >= implied growth) under a Student-t
  fitted to that company's own year-on-year FCF growth in LOG space, with
  `probability.observations` data points — typically 3. Three observations cannot
  support a precise estimate, so quote `probability.probability_range_pct` (the
  band implied by that sample size) rather than the point value.
- `probability.probability_pct` is null when the cash flows are too volatile for
  any estimate to mean anything; `probability.reason` explains it. Report that as
  "no usable estimate", never as a high or low probability.
- `beta_clamped`, `ke_clamped` and `beta_missing` flag where the discount rate
  rests on a boundary or a default rather than the company's own data.
- `risk_free_source` says whether the risk-free rate was a live market quote or a
  documented static assumption for that currency.

Valuation data:
{json.dumps(valoraciones, indent=2, ensure_ascii=False, default=str)}

Instructions:
1. For each company, state plainly what the price is assuming, whether the cash
   flows are trend-like enough for that to be compared with anything, and which
   way the gap cuts if so.
2. Separate the companies into those whose cash flows support a projection
   (`trend_r2` at or above 0.5) and those where the model declined. For the
   second group the honest output is the implied growth plus an explanation of
   why nothing further could be said — that is a real result, not a shortfall.
3. Interpret the probability honestly, as a range. A high band means this
   company's own history cleared that bar often; it says nothing about whether
   the future will.
4. Close with a section on where this model is most likely to be wrong. Cover at
   least: the three-observation sample behind every probability, the four-year
   FCF window and how badly it can misread a cyclical, the single 2.5% terminal
   growth applied across very different businesses, and the fact that the model
   sees only cash flow — never the reason behind it, so a legal settlement, a
   licensing payment or a deferred capex is indistinguishable from a trend.

Rules:
- Write the entire report in English.
- Only use the numbers above. Do not invent revenue, margins, guidance, segment
  detail or news. You have no search tool here and no other source.
- A negative `implied_growth_pct` means the price is assuming the business
  shrinks — say so explicitly rather than calling the stock "cheap".
- Never present a withheld figure as though it were low, high, or estimable. If
  `dcf_value_per_share` or `probability.probability_pct` is null, say what the
  model declined to compute and why.
- Do not rank on a gap computed against a trend whose `trend_r2` is below 0.5;
  such a gap is arithmetic, not evidence.
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
# RUNNER GENÉRICO — usado por cada screener.py de índice
# ==============================================================================
# Hay dos runners porque hay dos cribados, pero todo lo que va detrás del cribado
# — informe, valoración, escritura del JSON — es el mismo y vive en los helpers
# de aquí abajo. Para añadir un índice no hace falta tocar ninguno de los dos:
# basta un entry point que produzca su lista de tickers y llame al runner que le
# corresponda según lo que se quiera medir.
def _informe_seguro(generar_fn, etiqueta: str = "el informe") -> str | None:
    """Genera un informe de Claude sin que un fallo cueste la ejecución entera.

    El cribado ya tiene valor por sí solo. Si el agente falla se escribe
    igualmente el JSON con las empresas seleccionadas y report=None, en vez de
    perder la semana y dejar el portal con los datos de la anterior.
    """
    try:
        return generar_fn()
    except Exception as e:
        print(f"⚠️  El agente no pudo generar {etiqueta}: {type(e).__name__}: {e}")
        print("   Se guardan igualmente los resultados del cribado, sin informe cualitativo.")
        return None


def _etapa_valoracion(
    empresas_seleccionadas: list,
    universo_nombre: str,
    pausa_entre_tickers: float,
) -> tuple[list, list, str | None]:
    """Etapa de valoración (DCF inverso) sobre las empresas que pasaron el cribado.

    Va después del informe cualitativo y en su propio try: son dos preguntas
    independientes ("¿es buena?" / "¿está creciendo?" y "¿qué asume el precio?"),
    así que un fallo aquí no debe costar el informe que ya está escrito, ni al revés.
    """
    if not empresas_seleccionadas:
        return [], [], None

    try:
        valoraciones, valoraciones_fallidas = analizar_valoraciones(
            empresas_seleccionadas,
            pausa_entre_tickers=pausa_entre_tickers,
        )
        valuation_report = None
        if valoraciones:
            valuation_report = generar_informe_valoracion(valoraciones, universo_nombre)
        return valoraciones, valoraciones_fallidas, valuation_report
    except Exception as e:
        print(f"⚠️  La etapa de valoración falló: {type(e).__name__}: {e}")
        print("   Se guarda el JSON sin el bloque de valoración.")
        return [], [], None


def _metodo_valoracion() -> dict:
    """Documentación del modelo de valoración, embebida en cada JSON de salida.

    Vive en el JSON y no solo en el código porque el portal la renderiza tal cual:
    lo que el lector ve sobre los límites del modelo y lo que el modelo hace de
    verdad tienen que salir de la misma fuente.
    """
    return {
        "model": "Two-stage DCF on levered free cash flow (FCF discounted at cost of equity, compared with market cap — no net-debt bridge)",
        "horizon_years": valuation.HORIZONTE_ANIOS,
        "terminal_growth_pct": valuation.CRECIMIENTO_TERMINAL * 100,
        "discount_rate": f"CAPM: currency-matched risk-free rate + beta x {valuation.PRIMA_RIESGO_MERCADO * 100:.0f}% ERP, beta clamped to [{valuation.BETA_MIN}, {valuation.BETA_MAX}], rate clamped to [{valuation.KE_MIN * 100:.0f}%, {valuation.KE_MAX * 100:.0f}%]. Only USD has a live quote (^TNX); other currencies use a documented static rate, flagged per company in risk_free_source.",
        "implied_growth": "Reverse DCF — the FCF growth rate that sets the model's equity value equal to today's market cap. What the price assumes, not a forecast. Available for every company, since it needs no history.",
        "trend_test": f"A projection is only made when a least-squares line through log FCF reaches R2 >= {valuation.R2_MINIMO_PARA_PROYECTAR}. Below that the series is a path rather than a direction, dcf_value_per_share is null and dcf_skipped_reason says so.",
        "projected_growth_band_pct": [valuation.G_MODELADO_MIN * 100, valuation.G_MODELADO_MAX * 100],
        "probability": f"P(growth >= implied growth) under a Student-t fitted to the company's own year-on-year FCF growth in log space. Withheld entirely above a log stdev of {valuation.LOG_STDEV_MAX_PUBLICABLE}, where it would be indistinguishable from a coin flip.",
        "known_limits": [
            "History depth is uneven. SEC filings give US companies 14-17 years (16 growth observations for Adobe), but that source only covers SEC filers, so the IBEX screener and any company whose filing history has gaps falls back to yfinance's 4 years — 3 observations. Each company's fcf_source and probability.observations say which it got, and every probability is published as a range rather than a point.",
            "A short FCF window can catch a trough, a spike or both and mistake it for a trend, and a long one can fit an earlier version of the company beautifully. The R2 test guards the first, and projecting the lower of the long-run and recent trends guards the second. Neither is a cure.",
            "Terminal growth is a single 2.5% applied to every business regardless of its long-run ceiling; dcf_terminal_value_share_pct shows how much of each valuation rests on it.",
            "The model sees free cash flow only — no revenue detail, margins, guidance or segments. A one-off legal settlement, licensing payment or deferred capex is indistinguishable from a genuine trend change, which is why revenue growth is carried alongside as a corroboration check.",
            "Levered FCF is compared directly against market cap with no net-debt bridge — a deliberate simplification for data robustness across 500 tickers.",
        ],
    }


def _escribir_json(output_filename: str, output: dict) -> str:
    """Escribe el JSON de salida en data/ y devuelve la ruta."""
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Informe escrito en {output_path}")
    return output_path


def run_pipeline(
    obtener_tickers_fn,
    output_filename: str,
    universo_nombre: str,
    limite_analisis_default: int = 500,
    pausa_entre_tickers: float = 0.4,
    roa_minimo: float | None = 0.12,
) -> str:
    """Ejecuta el cribado de CALIDAD para un índice y escribe el JSON de salida.

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
        report_text = _informe_seguro(
            lambda: generar_informe(empresas_seleccionadas, universo_nombre, roa_minimo)
        )
    else:
        print("Ninguna empresa cumplió los filtros esta semana — se omite la llamada a Claude.")

    valoraciones, valoraciones_fallidas, valuation_report = _etapa_valoracion(
        empresas_seleccionadas, universo_nombre, pausa_entre_tickers
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(todos_los_tickers),
        "analyzed": min(limite_analisis, len(todos_los_tickers)),
        "passed_filters": len(empresas_seleccionadas),
        "companies": empresas_seleccionadas,
        "failed": tickers_fallidos,
        "report": report_text,
        "valuation_method": _metodo_valoracion(),
        "valuations": valoraciones,
        "valuation_failed": valoraciones_fallidas,
        "valuation_report": valuation_report,
    }

    return _escribir_json(output_filename, output)


def run_pipeline_crecimiento(
    obtener_tickers_fn,
    output_filename: str,
    universo_nombre: str,
    limite_analisis_default: int = 110,
    pausa_entre_tickers: float = 0.4,
) -> str:
    """Ejecuta el cribado de CRECIMIENTO + MOMENTUM y escribe el JSON de salida.

    Mismo esqueleto que run_pipeline — cribado, informe, valoración, JSON — con
    otro filtro y otro agente. La etapa de valoración se comparte sin cambios: el
    DCF inverso es justo la contrapregunta que le falta a un cribado de
    crecimiento, porque mide cuánto de ese crecimiento ya está en el precio.

    `obtener_tickers_fn` puede devolver una lista de tickers o una tupla
    (tickers, fuente). Lo segundo permite que un entry point con varias fuentes
    diga cuál acabó usando, y que eso llegue al JSON en vez de quedarse en los
    logs de Actions.

    Devuelve la ruta del archivo escrito.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("⚠️ La variable de entorno ANTHROPIC_API_KEY no está definida.")

    resultado = obtener_tickers_fn()
    if isinstance(resultado, tuple):
        todos_los_tickers, universo_fuente = resultado
    else:
        todos_los_tickers, universo_fuente = resultado, None

    limite_analisis = int(os.environ.get("SCREENER_LIMIT", str(limite_analisis_default)))
    empresas_seleccionadas, tickers_fallidos = filtrar_acciones_crecimiento(
        todos_los_tickers,
        limite_analisis=limite_analisis,
        pausa_entre_tickers=pausa_entre_tickers,
    )

    report_text = None
    if empresas_seleccionadas:
        report_text = _informe_seguro(
            lambda: generar_informe_crecimiento(empresas_seleccionadas, universo_nombre)
        )
    else:
        print("Ninguna empresa cumplió los filtros esta semana — se omite la llamada a Claude.")

    valoraciones, valoraciones_fallidas, valuation_report = _etapa_valoracion(
        empresas_seleccionadas, universo_nombre, pausa_entre_tickers
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(todos_los_tickers),
        "universe_source": universo_fuente,
        "analyzed": min(limite_analisis, len(todos_los_tickers)),
        "passed_filters": len(empresas_seleccionadas),
        "companies": empresas_seleccionadas,
        "failed": tickers_fallidos,
        "report": report_text,
        "criteria": {
            "screen": "Growth and momentum. No valuation, profitability or leverage filter.",
            "revenue_growth_min_pct": CRECIMIENTO_INGRESOS_MIN * 100,
            "earnings_growth_min_pct": CRECIMIENTO_BENEFICIOS_MIN * 100,
            "free_cash_flow": "Positive trailing free cash flow. The only quality guardrail in the screen — it separates growth that generates cash from growth that burns it.",
            "trend": "Price above the 50-day moving average, and the 50-day above the 200-day. The second is the underlying trend; the first alone can be a two-week bounce.",
            "return_6m": "Positive 6-month total return, on split- and dividend-adjusted prices.",
            "rsi_min": RSI_MINIMO_CRECIMIENTO,
            "excluded_on_purpose": "There is no P/E, ROE or debt filter. In this index a P/E under 20 rejects almost the whole universe, and what it lets through is the least representative of what the index is. A low multiple here does not signal quality; it signals that the market has stopped expecting growth.",
            "score": "Within-cohort percentile rank, averaged over revenue growth, earnings growth and 6-month return. 100 means best of the names that passed this week, not best company — a name's score moves when the rest of the cohort moves. Ranks rather than raw values, so that one company emerging from a cyclical trough with triple-digit earnings growth cannot dominate the average.",
        },
        "valuation_method": _metodo_valoracion(),
        "valuations": valoraciones,
        "valuation_failed": valoraciones_fallidas,
        "valuation_report": valuation_report,
    }

    return _escribir_json(output_filename, output)
