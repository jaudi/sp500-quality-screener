"""
Screener de fondos transparentes (ETFs UCITS) — pipeline agéntica con Groq.

A diferencia de screener.py / screener_ibex35.py (que filtran acciones
individuales de un índice), este script parte del catálogo público de
productos iShares, filtra por vehículo (solo ETFs), domicilio (IE/GB/LU) y
comisión (TER < 0.20%), calcula el Sharpe ratio de cada ETF a partir de su
histórico de precios de 3 años, y genera un comentario cualitativo con un
agente Groq sobre el top 10 resultante.

El resultado se escribe en data/latest-report-funds.json para ser consumido
por el portal Next.js (financeplots.com).
"""

from common import run_pipeline_fondos


def main():
    run_pipeline_fondos(
        output_filename="latest-report-funds.json",
        universo_nombre="ETFs iShares transparentes (IE/GB/LU, TER<0.20%)",
        domicilios_validos=("Ireland", "United Kingdom", "Luxembourg"),
        ter_max=0.20,
        top_n=10,
    )


if __name__ == "__main__":
    main()
