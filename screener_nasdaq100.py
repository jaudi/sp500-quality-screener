"""
Nasdaq-100 Growth Screener — pipeline agéntica con Claude.

A diferencia de screener.py / screener_ibex35.py, que criban por calidad y
múltiplo (ROE, P/E, deuda), este parte de un universo donde ese filtro no
discrimina: un P/E < 20 descarta casi todo el Nasdaq-100, y lo poco que deja
pasar es justo lo menos representativo del índice. Aquí se criba por crecimiento
de ingresos y beneficios, caja libre positiva, y una estructura de tendencia
completa (MA50, MA200, retorno a 6 meses, RSI). Ver
common.filtrar_acciones_crecimiento.

El resultado se escribe en data/latest-report-nasdaq100.json para ser consumido
por el portal Next.js (financeplots.com).
"""

import io

import pandas as pd
import requests

from common import run_pipeline_crecimiento

SLICKCHARTS_URL = "https://www.slickcharts.com/nasdaq100"

# Wikipedia servía la lista de componentes del S&P 500 y del IBEX 35, y es de
# donde la sacan los otros dos screeners. Para el Nasdaq-100 ya no: la tabla de
# componentes se retiró del artículo y no hay ninguna tabla de ~100 filas que
# raspar. Slickcharts la publica completa y con el peso de cada nombre en el
# índice, así que es la fuente primaria aquí.
#
# Son ~101-102 tickers, no 100: el índice cuenta compañías, pero varias cotizan
# con dos clases de acción (GOOGL y GOOG son la misma empresa), y ambas líneas
# son constituyentes.
NASDAQ100_FALLBACK = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "SPCX", "GOOG", "AVGO",
    "META", "TSLA", "MU", "WMT", "AMD", "ASML", "INTC", "CSCO",
    "PLTR", "COST", "LRCX", "AMAT", "NFLX", "ARM", "PANW", "TXN",
    "SNDK", "KLAC", "LIN", "MRVL", "CRWD", "AMGN", "TMUS", "QCOM",
    "STX", "PEP", "ADI", "GILD", "SHOP", "WDC", "BKNG", "VRTX",
    "ISRG", "FTNT", "SBUX", "PDD", "APP", "ADP", "CEG", "ABNB",
    "ADBE", "MELI", "CSX", "CMCSA", "DASH", "MAR", "INTU", "MNST",
    "LITE", "CTAS", "REGN", "MDLZ", "CDNS", "DDOG", "SNPS", "ROST",
    "WBD", "ORLY", "AEP", "PCAR", "HON", "NBIS", "MPWR", "NXPI",
    "TER", "BKR", "FANG", "FAST", "MSTR", "ALAB", "HONA", "CRWV",
    "XEL", "PYPL", "CCEP", "WDAY", "EXC", "ADSK", "KDP", "TRI",
    "PAYX", "FER", "MCHP", "TTWO", "RKLB", "IDXX", "AXON", "ROP",
    "ODFL", "ALNY", "DXCM", "KHC", "GEHC", "CPRT",
]
"""Lista fijada el 2026-09-12 desde Slickcharts, en orden de peso. Solo se usa
si el scraping falla."""


def obtener_tickers_nasdaq100() -> tuple[list, str]:
    """Devuelve (tickers, fuente) para el Nasdaq-100.

    Intenta Slickcharts y, si falla, cae a la lista fijada arriba. El fallback
    existe porque esto corre desatendido una vez por semana y raspar un tercero
    es la parte frágil de la cadena: perder la ejecución entera porque un sitio
    cambió el HTML sería peor que correr sobre un universo algo desactualizado.
    El Nasdaq-100 solo se reconstituye una vez al año (diciembre), más algún
    cambio suelto, así que la lista fijada envejece despacio — pero envejece, y
    por eso la fuente usada viaja hasta el JSON en `universe_source` en vez de
    quedarse en los logs.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(SLICKCHARTS_URL, headers=headers, timeout=45)
        response.raise_for_status()

        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            columnas = [str(c) for c in table.columns]
            if "Symbol" not in columnas:
                continue

            tickers = table["Symbol"].dropna().astype(str).str.strip().tolist()
            # Un cambio silencioso de formato puede devolver una tabla con la
            # columna correcta y dos filas. Si no hay un índice entero ahí
            # dentro, el fallback es mejor dato que lo que se acaba de leer.
            if len(tickers) < 90:
                raise ValueError(
                    f"Slickcharts devolvió solo {len(tickers)} símbolos; se esperaban ~100."
                )

            # Yahoo usa guiones donde el mercado usa puntos (BRK.B -> BRK-B).
            # Hoy el Nasdaq-100 no tiene ninguno, pero la normalización va aquí
            # igual que en screener.py, porque la composición cambia.
            tickers = [t.replace(".", "-") for t in tickers]
            print(f"✅ Se han extraído {len(tickers)} tickers del Nasdaq-100 desde Slickcharts.")
            return tickers, "slickcharts.com/nasdaq100 (live)"

        raise ValueError("No se encontró una tabla con columna 'Symbol' en Slickcharts.")

    except Exception as e:
        print(f"⚠️  No se pudo leer el universo desde Slickcharts: {type(e).__name__}: {e}")
        print(f"   Se usa la lista fijada en el repo ({len(NASDAQ100_FALLBACK)} tickers).")
        return list(NASDAQ100_FALLBACK), "pinned list in screener_nasdaq100.py (Slickcharts unreachable)"


def main():
    run_pipeline_crecimiento(
        obtener_tickers_fn=obtener_tickers_nasdaq100,
        output_filename="latest-report-nasdaq100.json",
        universo_nombre="the Nasdaq-100",
        # Por encima de los ~102 constituyentes, para que un alta en el índice no
        # quede fuera del análisis por el límite en vez de por los filtros.
        limite_analisis_default=110,
    )


if __name__ == "__main__":
    main()
