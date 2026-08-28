"""
IBEX 35 Quality Screener — pipeline agéntica con Groq.

Extrae los tickers del IBEX 35 desde Wikipedia, aplica el mismo cribado
fundamental + técnico que el screener del S&P 500 (ver common.py), y genera
un informe cualitativo con Groq. El resultado se escribe en
data/latest-report-ibex35.json para ser consumido por el portal Next.js
(financeplots.com).
"""

import io

import pandas as pd
import requests

from common import run_pipeline


def obtener_tickers_ibex35() -> list:
    """Extrae los tickers del IBEX 35 desde Wikipedia (ya incluyen el sufijo .MC de yfinance)."""
    url = "https://en.wikipedia.org/wiki/IBEX_35"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    for table in tables:
        columnas = [str(c) for c in table.columns]
        if "Ticker" in columnas and "Company" in columnas:
            tickers = table["Ticker"].dropna().astype(str).tolist()
            print(f"✅ Se han extraído {len(tickers)} tickers del IBEX 35 desde Wikipedia.")
            return tickers

    raise ValueError("No se encontró la tabla de componentes del IBEX 35 en Wikipedia.")


def main():
    run_pipeline(
        obtener_tickers_fn=obtener_tickers_ibex35,
        output_filename="latest-report-ibex35.json",
        universo_nombre="el IBEX 35",
        limite_analisis_default=35,
    )


if __name__ == "__main__":
    main()
