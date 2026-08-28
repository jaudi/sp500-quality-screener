"""
S&P 500 Quality Screener — pipeline agéntica con Groq.

Extrae tickers del S&P 500, aplica un cribado fundamental + técnico,
y genera un informe cualitativo con un agente Groq (tool-use + búsqueda web).
El resultado se escribe en data/latest-report.json para ser consumido por
el portal Next.js (financeplots.com).
"""

import io

import pandas as pd
import requests

from common import run_pipeline


def obtener_tickers_sp500() -> list:
    """Extrae y normaliza los tickers del S&P 500 desde Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]

    # Reemplazar puntos por guiones para yfinance (ej. BRK.B -> BRK-B)
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"✅ Se han extraído {len(tickers)} tickers del S&P 500 desde Wikipedia.")
    return tickers


def main():
    run_pipeline(
        obtener_tickers_fn=obtener_tickers_sp500,
        output_filename="latest-report.json",
        universo_nombre="el S&P 500",
        limite_analisis_default=500,
        roa_minimo=0.12,  # explícito: el S&P 500 sí exige ROA como filtro duro
    )


if __name__ == "__main__":
    main()
