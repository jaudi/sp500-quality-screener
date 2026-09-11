"""
Histórico largo de flujo de caja desde los XBRL de la SEC.

yfinance solo expone 4-5 ejercicios anuales, que es la raíz de casi todas las
debilidades del modelo de valoración: con 3 crecimientos interanuales, ni la
probabilidad ni el test de R² tienen de dónde agarrarse, y una ventana de 4 años
sobre una cíclica puede caer entera dentro de un suelo o de un pico.

La API pública de la SEC (data.sec.gov, sin clave) devuelve todo lo que una
empresa ha declarado alguna vez para un concepto contable. Para Adobe son 17
ejercicios en vez de 4. Eso convierte la muestra de 3 observaciones en ~16.

Solo cubre a quien presenta ante la SEC, o sea el S&P 500. El IBEX se queda con
yfinance y con su ventana corta — la diferencia de fiabilidad entre los dos
screeners es real y se refleja en fcf_source, no se disimula.
"""

import json
import time

import requests

# La SEC exige un User-Agent que identifique a quien llama y pide no pasar de
# 10 peticiones por segundo. Con 5 empresas por ejecución no nos acercamos, pero
# la pausa se respeta igualmente por si algún día se amplía el cribado.
SEC_USER_AGENT = "FinancePlots screener (contact: javier.audibert@gmail.com)"
SEC_PAUSA = 0.12
TIMEOUT = 20

URL_TICKERS = "https://www.sec.gov/files/company_tickers.json"
URL_CONCEPTO = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concepto}.json"

# Varias empresas usan etiquetas distintas para la misma línea. Se prueban en
# orden y se coge la primera que devuelva algo utilizable.
CONCEPTOS_FLUJO_OPERATIVO = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CONCEPTOS_CAPEX = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
)

# Un histórico que termina hace años describe otra etapa de la empresa: más
# puntos, pero peores. Por encima de esto se prefiere la ventana corta de
# yfinance, que al menos llega hasta hoy.
ANTIGUEDAD_MAXIMA_ANIOS = 2

# Fracción de ejercicios que tiene que estar presente entre el primero y el
# último. Con huecos, los índices del ajuste log-lineal dejan de estar
# igualmente espaciados y la pendiente ya no es un crecimiento anual.
COBERTURA_MINIMA = 0.8

_cache_tickers: dict | None = None
_sesion: requests.Session | None = None


def _cliente() -> requests.Session:
    global _sesion
    if _sesion is None:
        _sesion = requests.Session()
        _sesion.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return _sesion


def obtener_cik(ticker: str) -> int | None:
    """Traduce ticker a CIK usando el índice público de la SEC (se cachea en memoria)."""
    global _cache_tickers

    # Los sufijos de mercado (.MC del IBEX, .L de Londres) no existen en la SEC.
    if "." in ticker:
        return None

    if _cache_tickers is None:
        try:
            respuesta = _cliente().get(URL_TICKERS, timeout=TIMEOUT)
            respuesta.raise_for_status()
            crudo = respuesta.json()
            _cache_tickers = {
                str(fila["ticker"]).upper(): int(fila["cik_str"]) for fila in crudo.values()
            }
        except Exception:
            _cache_tickers = {}

    # yfinance usa guion donde la SEC usa punto (BRK-B frente a BRK.B).
    return _cache_tickers.get(ticker.upper()) or _cache_tickers.get(ticker.upper().replace("-", "."))


def _serie_anual_concepto(cik: int, conceptos: tuple[str, ...]) -> dict[int, float]:
    """{ejercicio: valor} uniendo TODAS las etiquetas equivalentes del concepto.

    Quedarse con la primera etiqueta que devolviera algo parecía suficiente y no
    lo era: las empresas cambian de etiqueta a mitad de su historia. Incyte
    declara el capex bajo PaymentsToAcquirePropertyPlantAndEquipment hasta 2016 y
    bajo otra a partir de ahí, así que la primera etiqueta daba una serie que se
    cortaba en 2016 — seis años, de los cuales solo uno servía, y un R² de 0,999
    ajustado sobre casi nada. Uniendo las etiquetas la historia queda completa.
    """
    por_ejercicio: dict[int, float] = {}

    for concepto in conceptos:
        try:
            respuesta = _cliente().get(URL_CONCEPTO.format(cik=cik, concepto=concepto), timeout=TIMEOUT)
            time.sleep(SEC_PAUSA)
            if respuesta.status_code != 200:
                continue
            unidades = respuesta.json().get("units", {}).get("USD", [])
        except Exception:
            continue

        for fila in unidades:
            # Solo 10-K y solo periodos de ejercicio completo: los XBRL mezclan
            # trimestres, acumulados y reexpresiones bajo el mismo concepto, y
            # sumarlos sin filtrar da cifras que no existen en ningún estado.
            if fila.get("form") != "10-K" or fila.get("fp") != "FY":
                continue
            inicio, fin = fila.get("start"), fila.get("end")
            ejercicio = fila.get("fy")
            if not (inicio and fin and ejercicio):
                continue
            dias = (_fecha(fin) - _fecha(inicio)).days
            if not (350 <= dias <= 380):
                continue
            # Una reexpresión posterior del mismo ejercicio pisa a la anterior,
            # que es lo que queremos: la última versión declarada.
            por_ejercicio[int(ejercicio)] = float(fila["val"])

    return por_ejercicio


def _fecha(iso: str):
    from datetime import date

    return date.fromisoformat(iso)


def serie_free_cash_flow_sec(ticker: str) -> tuple[list[float], str]:
    """FCF anual (flujo operativo - capex) desde la SEC, de más antiguo a más reciente.

    Devuelve ([], "") cuando la empresa no presenta ante la SEC o los datos no
    alcanzan, para que el llamante caiga a yfinance sin tener que distinguir
    entre "no aplica" y "ha fallado".
    """
    cik = obtener_cik(ticker)
    if cik is None:
        return [], ""

    flujo = _serie_anual_concepto(cik, CONCEPTOS_FLUJO_OPERATIVO)
    if len(flujo) < 3:
        return [], ""

    capex = _serie_anual_concepto(cik, CONCEPTOS_CAPEX)

    # Solo ejercicios con ambas patas: restar un capex ausente como si fuera
    # cero convertiría el flujo operativo en "free cash flow" y exageraría el
    # crecimiento justo en los años peor documentados.
    ejercicios = sorted(set(flujo) & set(capex)) if capex else []
    if len(ejercicios) < 3:
        return [], ""

    # Una serie que se corta hace años es peor que la ventana corta de yfinance:
    # tiene más puntos pero describe otra época de la empresa. Si el último
    # ejercicio disponible no es reciente, se cede el paso.
    from datetime import date

    if date.today().year - ejercicios[-1] > ANTIGUEDAD_MAXIMA_ANIOS:
        return [], ""

    # Tampoco vale una historia con agujeros: si faltan ejercicios intermedios,
    # los índices del ajuste dejan de representar el paso del tiempo y la recta
    # mide otra cosa.
    esperados = ejercicios[-1] - ejercicios[0] + 1
    if len(ejercicios) < esperados * COBERTURA_MINIMA:
        return [], ""

    # El capex viene declarado en positivo (es un pago), al revés que en yfinance.
    serie = [flujo[e] - abs(capex[e]) for e in ejercicios]
    return serie, f"SEC XBRL 10-K filings ({ejercicios[0]}-{ejercicios[-1]})"
