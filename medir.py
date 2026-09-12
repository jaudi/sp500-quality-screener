"""
Medidor de rendimiento forward — ¿lo que eligieron los screeners batió al índice?

Es la única pieza del repo que mira hacia atrás. Todo lo demás responde "¿qué
comprar?"; esto responde "¿y qué pasó después?", que es la pregunta que convierte
la opinión en evidencia.

No necesita datos nuevos. Cada informe commiteado ya guarda `precio_actual` de
cada empresa y git guarda la fecha del commit, así que el precio y la fecha de
entrada llevan recogiéndose desde la primera ejecución sin que nadie lo
planease. Este script solo los lee.

El resultado se escribe en data/performance.json.

Cosas que morderán:

- **Necesita el historial completo de git.** `actions/checkout` clona a
  profundidad 1 por defecto, y con un solo commit este script no mide nada y no
  falla — devuelve una entrada por ticker con fecha de hoy y un 0% perfecto, que
  es peor que un error porque parece un resultado. El workflow pasa
  `fetch-depth: 0` por eso.
- **La entrada es la primera aparición del ticker, y se mantiene para siempre.**
  Es una decisión, no una verdad: los screeners no tienen regla de venta, así
  que hay que elegir una para poder medir. Buy-and-hold desde la primera señal
  es la más simple y la más difícil de manipular a posteriori. `still_passing`
  dice si el nombre sigue pasando el filtro hoy, que es donde se vería si haría
  falta una.
- **La comparación con el índice es el resultado, no un adorno.** Una cartera al
  -3% es excelente si el índice cayó un 10% y mala si subió un 6%. El retorno
  suelto no significa nada.
- **Con pocas semanas esto es ruido.** El JSON lleva `weeks_of_history` y
  `n_positions` para que quien lo lea pueda ver que todavía no dice nada.
"""

import json
import os
import subprocess
import warnings
from datetime import datetime, timezone

import yfinance as yf

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# fichero del informe, nombre del índice de referencia
SCREENERS = {
    "sp500": ("data/latest-report.json", "^GSPC", "S&P 500"),
    "ibex35": ("data/latest-report-ibex35.json", "^IBEX", "IBEX 35"),
    "nasdaq100": ("data/latest-report-nasdaq100.json", "^NDX", "Nasdaq-100"),
}


def _git(*args) -> str:
    return subprocess.run(
        ["git", "-C", REPO_DIR, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def historial_de_entradas(fichero: str) -> dict:
    """{ticker: {entry_date, entry_price}} — la primera vez que el screener lo eligió.

    Recorre los commits del informe de más antiguo a más reciente y se queda con
    la primera aparición de cada ticker. Un nombre que entra, sale y vuelve a
    entrar conserva su entrada original: haber acertado dos veces con la misma
    empresa no debería contar como dos ideas.
    """
    lineas = _git("log", "--format=%H %ad", "--date=short", "--", fichero).splitlines()
    commits = [l.split() for l in lineas if l.strip()]

    entradas = {}
    for commit_hash, fecha in reversed(commits):
        crudo = _git("show", f"{commit_hash}:{fichero}")
        if not crudo.strip():
            continue
        try:
            informe = json.loads(crudo)
        except json.JSONDecodeError:
            # Un informe a medio escribir en un commit antiguo no debe costar la medición.
            continue
        for empresa in informe.get("companies", []):
            ticker = empresa.get("ticker")
            precio = empresa.get("precio_actual")
            if ticker and precio and ticker not in entradas:
                entradas[ticker] = {"entry_date": fecha, "entry_price": float(precio)}

    return entradas


def precios_actuales(tickers: list) -> dict:
    """Último cierre de cada ticker. Descarga en bloque: son decenas, no cientos."""
    if not tickers:
        return {}
    datos = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False, threads=True)
    if datos.empty:
        return {}

    cierres = datos["Close"]
    precios = {}
    for t in tickers:
        try:
            serie = cierres[t] if len(tickers) > 1 else cierres
            serie = serie.dropna()
            if len(serie):
                precios[t] = float(serie.iloc[-1])
        except (KeyError, IndexError):
            continue
    return precios


def retorno_indice(benchmark: str, desde: str) -> float | None:
    """Retorno del índice desde la fecha de la primera entrada, misma ventana."""
    try:
        hist = yf.Ticker(benchmark).history(start=desde)
        if hist.empty or len(hist) < 2:
            return None
        cierre = hist["Close"]
        return round((float(cierre.iloc[-1]) / float(cierre.iloc[0]) - 1) * 100, 2)
    except Exception as e:
        print(f"   ⚠️  no se pudo leer el índice {benchmark}: {type(e).__name__}: {e}")
        return None


def medir(clave: str, fichero: str, benchmark: str, nombre: str) -> dict | None:
    print(f"\n{'=' * 70}\n📈 {nombre}\n{'=' * 70}")

    entradas = historial_de_entradas(fichero)
    if not entradas:
        print("   sin historial todavía — se omite")
        return None

    # Quién sigue pasando el filtro hoy. Un nombre que salió y luego cayó mucho
    # es justo el caso que dice si hace falta una regla de venta.
    try:
        with open(os.path.join(REPO_DIR, fichero), encoding="utf-8") as f:
            vigentes = {c["ticker"] for c in json.load(f).get("companies", [])}
    except (OSError, json.JSONDecodeError):
        vigentes = set()

    precios = precios_actuales(sorted(entradas))
    fecha_min = min(e["entry_date"] for e in entradas.values())

    posiciones = []
    for ticker, entrada in sorted(entradas.items(), key=lambda kv: (kv[1]["entry_date"], kv[0])):
        actual = precios.get(ticker)
        retorno = round((actual / entrada["entry_price"] - 1) * 100, 2) if actual else None
        posiciones.append(
            {
                "ticker": ticker,
                "entry_date": entrada["entry_date"],
                "entry_price": round(entrada["entry_price"], 2),
                "current_price": round(actual, 2) if actual else None,
                "return_pct": retorno,
                "still_passing": ticker in vigentes,
            }
        )
        estado = "" if ticker in vigentes else "  (ya no pasa el filtro)"
        print(
            f"   {ticker:8} {entrada['entry_date']}  "
            f"{entrada['entry_price']:9.2f} → {actual if actual else 0:9.2f}  "
            f"{retorno if retorno is not None else 0:+7.2f}%{estado}"
        )

    medidos = [p["return_pct"] for p in posiciones if p["return_pct"] is not None]
    if not medidos:
        print("   ningún precio actual disponible — se omite")
        return None

    cartera = round(sum(medidos) / len(medidos), 2)
    indice = retorno_indice(benchmark, fecha_min)
    alfa = round(cartera - indice, 2) if indice is not None else None

    semanas = round((datetime.now(timezone.utc).date() - datetime.strptime(fecha_min, "%Y-%m-%d").date()).days / 7, 1)

    print(f"   {'-' * 60}")
    print(f"   CARTERA  {cartera:+7.2f}%   n={len(medidos)}   ({semanas} semanas)")
    if indice is not None:
        print(f"   {benchmark:8} {indice:+7.2f}%")
        print(f"   ALFA     {alfa:+7.2f} pp")

    return {
        "name": nombre,
        "benchmark": benchmark,
        "since": fecha_min,
        "weeks_of_history": semanas,
        "n_positions": len(medidos),
        "portfolio_return_pct": cartera,
        "benchmark_return_pct": indice,
        "alpha_pp": alfa,
        "positions": posiciones,
    }


def main():
    # Un clon superficial produce una medición falsa, no un error: una sola
    # entrada por ticker fechada hoy y un 0% impecable. Mejor parar.
    profundidad = len([l for l in _git("log", "--format=%H").splitlines() if l.strip()])
    if profundidad <= 1:
        raise SystemExit(
            "⚠️  El repo parece un clon superficial (1 commit). El medidor necesita "
            "el historial completo — usa `fetch-depth: 0` en actions/checkout."
        )

    resultados = {}
    for clave, (fichero, benchmark, nombre) in SCREENERS.items():
        medida = medir(clave, fichero, benchmark, nombre)
        if medida:
            resultados[clave] = medida

    salida = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "entry": "First time the screener selected the name. The price recorded in that run's report is the entry price, and the commit date is the entry date.",
            "holding": "Buy and hold from that first signal, equally weighted, never sold. The screeners have no sell rule, so one had to be chosen to make measurement possible — this is the simplest and the hardest to tune after the fact. `still_passing` shows which names would already have left a filter-based rule.",
            "benchmark": "The index over the same window, from the earliest entry date. The portfolio return on its own says nothing: -3% is excellent against an index that fell 10% and poor against one that rose 6%. Alpha is the result.",
            "costs": "No transaction costs, spreads, taxes or dividends. Gross price return only.",
            "caveat": "Read weeks_of_history and n_positions before reading anything else. Over a few weeks and a handful of names this is noise, not evidence.",
        },
        "screeners": resultados,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    ruta = os.path.join(DATA_DIR, "performance.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Escrito en {ruta}")


if __name__ == "__main__":
    main()
