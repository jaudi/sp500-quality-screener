"""
Prueba de que la recogida produce TODAS las métricas que los factores declaran.

A diferencia de `test_factores.py`, esta sí usa la red: descarga un ticker real.
Sigue sin gastar tokens — no toca Claude.

Existe por un fallo que ya ha aparecido tres veces: una métrica declarada en
`FACTORES` que `recolectar_universo` nunca produce. No rompe nada, no da error, y
no se ve en la tabla — el factor simplemente deja de pesar en silencio y el score
publicado no es el que dice la metodología. Pasó con los devengos (el
denominador vivía en el balance, no en `info`), y con ROIC y EV/EBIT, que estaban
declarados desde el primer commit del motor sin que nadie los calculara.

Se ejecuta con `python test_recoleccion.py`.
"""

import warnings

warnings.filterwarnings("ignore")

import common
import factores

# Uno grande y líquido, con cuentas completas: si falta algo aquí, falta siempre.
TICKER = "MSFT"

print(f"\nRecogiendo {TICKER}...")
universo, fallidos = common.recolectar_universo([TICKER], pausa_entre_tickers=0)

fallos = []

if not universo:
    raise SystemExit(f"❌ No se pudo recoger {TICKER}: {fallidos}")

fila = universo[0]
declaradas = [m.clave for metricas in factores.FACTORES.values() for m in metricas]

print(f"\n{len(declaradas)} métricas declaradas en FACTORES\n")

for clave in declaradas:
    if clave not in fila:
        print(f"  ❌ {clave:36} AUSENTE — declarada pero nunca recogida")
        fallos.append(clave)
    elif fila[clave] is None:
        # `exceso_implicito_pp` es legítimamente None aquí: se rellena en la
        # segunda etapa, después del DCF inverso. El resto no tiene excusa en un
        # ticker como este.
        if clave == "exceso_implicito_pp":
            print(f"  ⏭️  {clave:36} None (se rellena tras la valoración)")
        else:
            print(f"  ⚠️  {clave:36} None — la clave existe pero no se calculó")
            fallos.append(clave)
    else:
        print(f"  ✅ {clave:36} {fila[clave]}")

# La cobertura que se publica junto al score tiene que reflejar la realidad.
factores.puntuar_factores(universo, factores.PESOS["sp500"])
cobertura = universo[0]["cobertura_pct"]
esperada = round((len(declaradas) - 1) / len(declaradas) * 100)  # todas menos expectativas
print(f"\ncobertura_pct publicada: {cobertura}%  (esperada ~{esperada}% sin expectativas)")
if cobertura < esperada:
    print("  ❌ la cobertura no llega a lo esperado")
    fallos.append("cobertura_pct")

print()
if fallos:
    raise SystemExit(f"❌ {len(fallos)} métrica(s) sin recoger: {', '.join(fallos)}")
print("✅ Todas las métricas declaradas se recogen.")
