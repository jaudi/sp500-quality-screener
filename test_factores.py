"""
Pruebas del motor multifactor. Sin red, sin tokens: aritmética contra invariantes.

Se ejecuta con `python test_factores.py`. No hay pytest en requirements.txt y no
merece la pena añadirlo para esto.
"""

from factores import FACTORES, PESOS, puntuar_factores, rango_percentil


fallos = []


def comprobar(condicion, mensaje):
    if condicion:
        print(f"  ✅ {mensaje}")
    else:
        print(f"  ❌ {mensaje}")
        fallos.append(mensaje)


print("\nrango_percentil")

r = rango_percentil([1, 2, 3, 4, 5], mayor_es_mejor=True)
comprobar(r == [0.0, 25.0, 50.0, 75.0, 100.0], f"escala 0-100 uniforme → {r}")

r = rango_percentil([1, 2, 3, 4, 5], mayor_es_mejor=False)
comprobar(r == [100.0, 75.0, 50.0, 25.0, 0.0], f"mayor_es_mejor=False invierte → {r}")

r = rango_percentil([10, 20, None, 40], mayor_es_mejor=True)
comprobar(r[2] is None, "un None sale como None, no como 0")
comprobar(r[0] == 0.0 and r[3] == 100.0, "los presentes se rankean entre ellos")

r = rango_percentil([5, 5, 5, 9], mayor_es_mejor=True)
comprobar(r[0] == r[1] == r[2], f"los empates comparten rango → {r}")
comprobar(r[3] == 100.0, "el valor único mayor se lleva el 100")

comprobar(rango_percentil([]) == [], "lista vacía no revienta")
comprobar(rango_percentil([None, None]) == [None, None], "todo None no revienta")
comprobar(rango_percentil([7]) == [50.0], "un solo valor es mediano, no perfecto")

# El caso Micron: el rango tiene que tratarlo como "el mejor", no como 1368
# unidades. Si esto falla, el score vuelve a ser una columna disfrazada de tres.
r = rango_percentil([5.0, 8.0, 12.0, 1368.0], mayor_es_mejor=True)
comprobar(r == [0.0, 33.33, 66.67, 100.0], f"un outlier extremo no distorsiona la escala → {r}")


print("\npuntuar_factores")

universo = [
    # barata, calidad intermedia, sin momentum ni crecimiento
    {"ticker": "VALOR", "per_normalizado": 8, "precio_valor_libros": 0.9, "fcf_yield": 11.0,
     "ev_ebit": 6, "roic": 19, "margen_operativo": 23, "conversion_fcf": 1.05, "devengos": 0.02,
     "deuda_neta_ebitda": 1.0, "cobertura_intereses": 13, "retorno_6m": -12.0,
     "retorno_12m": -8.0, "distancia_ma200_pct": -9.0, "exceso_implicito_pp": -4.0,
     "crecimiento_ingresos_normalizado": 1.0, "crecimiento_beneficios_normalizado": 2.0},
    # cara, pero buen negocio que crece — el perfil tipico del Nasdaq
    {"ticker": "MOMENTO", "per_normalizado": 41, "precio_valor_libros": 9.5, "fcf_yield": 1.4,
     "ev_ebit": 33, "roic": 26, "margen_operativo": 31, "conversion_fcf": 1.2, "devengos": 0.005,
     "deuda_neta_ebitda": 0.3, "cobertura_intereses": 22, "retorno_6m": 61.0,
     "retorno_12m": 88.0, "distancia_ma200_pct": 27.0, "exceso_implicito_pp": 32.0,
     "crecimiento_ingresos_normalizado": 48.0, "crecimiento_beneficios_normalizado": 55.0},
    # mediana en todo
    {"ticker": "MEDIA", "per_normalizado": 17, "precio_valor_libros": 2.6, "fcf_yield": 5.2,
     "ev_ebit": 14, "roic": 15, "margen_operativo": 19, "conversion_fcf": 0.95, "devengos": 0.04,
     "deuda_neta_ebitda": 1.6, "cobertura_intereses": 9, "retorno_6m": 7.0,
     "retorno_12m": 12.0, "distancia_ma200_pct": 4.0, "exceso_implicito_pp": 6.0,
     "crecimiento_ingresos_normalizado": 12.0, "crecimiento_beneficios_normalizado": 14.0},
]

sp = puntuar_factores([dict(e) for e in universo], PESOS["sp500"])
por_ticker = {e["ticker"]: e for e in sp}

comprobar(por_ticker["VALOR"]["factores"]["value"] == 100.0, "VALOR se lleva el 100 en value")
comprobar(por_ticker["MOMENTO"]["factores"]["value"] == 0.0, "MOMENTO se lleva el 0 en value")
comprobar(por_ticker["MOMENTO"]["factores"]["momentum"] == 100.0, "MOMENTO se lleva el 100 en momentum")
comprobar(
    por_ticker["VALOR"]["factores"]["expectativas"] == 100.0,
    "expectativas premia al que el precio exige menos",
)
comprobar(sp[0]["ticker"] == "VALOR", f"con pesos S&P gana el barato → {[e['ticker'] for e in sp]}")

# La misma cohorte con los pesos del Nasdaq tiene que ordenar distinto: si no,
# los pesos no están haciendo nada.
nd = puntuar_factores([dict(e) for e in universo], PESOS["nasdaq100"])
comprobar(
    [e["ticker"] for e in nd] != [e["ticker"] for e in sp],
    f"los mismos datos se ordenan distinto segun los pesos → S&P {[e['ticker'] for e in sp]} vs Nasdaq {[e['ticker'] for e in nd]}",
)
comprobar(
    nd[0]["ticker"] == "MOMENTO",
    f"con pesos Nasdaq gana el de crecimiento → {[e['ticker'] for e in nd]}",
)

comprobar(all(0 <= e["score"] <= 100 for e in sp), "todos los scores caen en 0-100")
comprobar(all(e["cobertura_pct"] == 100 for e in sp), "cobertura 100% con datos completos")


print("\ndatos incompletos")

parcial = [
    dict(universo[0]),
    {k: v for k, v in universo[1].items() if k in ("ticker", "per_normalizado", "retorno_6m")},
    dict(universo[2]),
]
p = puntuar_factores(parcial, PESOS["sp500"])
manco = next(e for e in p if e["ticker"] == "MOMENTO")

comprobar(manco["score"] is not None, "una empresa con huecos sigue puntuando")
comprobar(manco["cobertura_pct"] < 100, f"la cobertura lo delata → {manco['cobertura_pct']}%")
comprobar("quality" not in manco["factores"], "un factor sin ninguna métrica no se inventa")
comprobar(
    "expectativas" not in manco["factores"],
    "un factor sin dato no se rellena con la mediana",
)

vacio = puntuar_factores([{"ticker": "NADA"}], PESOS["sp500"])
comprobar(vacio[0]["score"] is None, "sin ninguna métrica, score es None y no 0")

p2 = puntuar_factores([{"ticker": "NADA"}, dict(universo[0])], PESOS["sp500"])
comprobar(p2[0]["ticker"] == "VALOR", "las empresas sin score van al final, no al principio")

comprobar(puntuar_factores([], PESOS["sp500"]) == [], "universo vacío no revienta")


print("\nconfiguración")

for nombre, pesos in PESOS.items():
    comprobar(sum(pesos.values()) == 100, f"{nombre}: los pesos suman 100")
    comprobar(all(f in FACTORES for f in pesos), f"{nombre}: todos sus factores existen")

claves = [m.clave for fs in FACTORES.values() for m in fs]
comprobar(len(claves) == len(set(claves)), "ninguna métrica se repite entre factores")


print()
if fallos:
    raise SystemExit(f"❌ {len(fallos)} prueba(s) fallida(s)")
print("✅ Todo correcto.")
