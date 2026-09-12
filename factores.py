"""
Motor multifactor — puntúa un universo por factores en vez de filtrarlo por umbrales.

Es aritmética pura sobre diccionarios: no descarga nada, no llama a Claude, no
escribe ficheros. Eso es deliberado — igual que `valuation.py`, se puede probar
entero sin red y sin gastar un token.

## Por qué factores y no filtros

El cribado actual es una cadena de condiciones duras: ROE > 20% Y P/E < 20 Y
RSI > 30. Una empresa con ROE del 19,8% desaparece igual que una con ROE del 3%.
Eso tiene cuatro problemas:

1. Los umbrales son inventados y nunca se han validado.
2. La pantalla se vacía — el IBEX, con 35 valores y 5 filtros, devuelve 3 nombres.
3. Se tira la información del margen: cuánto pasó o falló cada empresa.
4. No hay forma de ponderar. Hoy el RSI pesa igual que el ROE sin que nadie lo
   haya decidido.

Un rango percentil transversal resuelve los cuatro: no necesita umbral, siempre
devuelve un orden, conserva el margen y admite pesos explícitos.

## Por qué rangos y no valores

Micron salió de un suelo cíclico con +1.368% de crecimiento de beneficios. En una
media de valores eso domina cualquier otra columna y convierte un score de tres
componentes en uno solo disfrazado. El rango percentil lo trata como lo que es
—el mejor de la cohorte, un puesto— y no como 1.368 unidades de nada.

## Qué NO hace este módulo

No decide qué comprar. Ordena. Los filtros duros siguen existiendo, pero sólo
para lo que impide analizar (sin precio, sin cuentas, FCF negativo donde la
valoración lo necesita), nunca para lo que puntúa bajo. Un filtro elimina lo
inevaluable; un factor ordena lo evaluable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrica:
    """Una columna del universo y en qué dirección es buena.

    `mayor_es_mejor=False` invierte el rango: en P/E, P/B, deuda o devengos, el
    percentil alto lo merece el valor bajo.
    """

    clave: str
    mayor_es_mejor: bool


# Composición de cada factor. Una métrica puede aparecer en varios factores si
# de verdad informa a los dos, pero conviene que no: duplicarla es ponderarla
# dos veces por la puerta de atrás.
FACTORES: dict[str, tuple[Metrica, ...]] = {
    "value": (
        Metrica("per_normalizado", mayor_es_mejor=False),
        Metrica("precio_valor_libros", mayor_es_mejor=False),
        Metrica("fcf_yield", mayor_es_mejor=True),
        Metrica("ev_ebit", mayor_es_mejor=False),
    ),
    "quality": (
        Metrica("roic", mayor_es_mejor=True),
        Metrica("margen_operativo", mayor_es_mejor=True),
        Metrica("conversion_fcf", mayor_es_mejor=True),
        Metrica("devengos", mayor_es_mejor=False),
        Metrica("deuda_neta_ebitda", mayor_es_mejor=False),
        Metrica("cobertura_intereses", mayor_es_mejor=True),
    ),
    "growth": (
        Metrica("crecimiento_ingresos_normalizado", mayor_es_mejor=True),
        Metrica("crecimiento_beneficios_normalizado", mayor_es_mejor=True),
    ),
    "momentum": (
        Metrica("retorno_6m", mayor_es_mejor=True),
        Metrica("retorno_12m", mayor_es_mejor=True),
        Metrica("distancia_ma200_pct", mayor_es_mejor=True),
    ),
    # La aportación propia: el DCF inverso ya calcula cuánto crecimiento exige el
    # precio. Aquí penaliza a quien cotiza pidiendo mucho más de lo que el
    # negocio ha entregado nunca. Es la única columna que mira a las expectativas
    # del mercado en vez de a la empresa.
    "expectativas": (
        Metrica("exceso_implicito_pp", mayor_es_mejor=False),
    ),
}

# Los pesos son la tesis de cada screener. Que estén aquí y no repartidos por el
# código es lo que permite cambiarlos y medir el efecto.
PESOS: dict[str, dict[str, float]] = {
    "sp500": {"value": 35, "quality": 35, "momentum": 15, "expectativas": 15},
    "ibex35": {"value": 40, "quality": 35, "momentum": 10, "expectativas": 15},
    "nasdaq100": {"growth": 40, "quality": 20, "momentum": 25, "expectativas": 15},
}


def rango_percentil(valores: list, mayor_es_mejor: bool = True) -> list:
    """Rango percentil 0-100 de cada valor dentro de la lista.

    Los None pasan como None: una empresa sin el dato no puntúa esa métrica, y
    sobre todo **no puntúa cero**. Tratar un hueco como el peor valor posible
    castiga a la empresa por un fallo de yfinance, que es la clase de error que
    no se ve en la tabla y decide el orden igualmente.

    Los empates reciben el rango medio del grupo, que es el comportamiento
    estándar y evita que el orden alfabético desempate por la puerta de atrás.
    """
    presentes = [(i, v) for i, v in enumerate(valores) if v is not None]
    if not presentes:
        return [None] * len(valores)
    if len(presentes) == 1:
        salida = [None] * len(valores)
        salida[presentes[0][0]] = 50.0
        return salida

    ordenados = sorted(presentes, key=lambda iv: iv[1], reverse=not mayor_es_mejor)

    salida = [None] * len(valores)
    n = len(ordenados)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordenados[j + 1][1] == ordenados[i][1]:
            j += 1
        # Rango medio del grupo empatado, escalado a 0-100.
        rango_medio = (i + j) / 2
        percentil = round(rango_medio / (n - 1) * 100, 2)
        for k in range(i, j + 1):
            salida[ordenados[k][0]] = percentil
        i = j + 1
    return salida


def puntuar_factores(universo: list, pesos: dict) -> list:
    """Añade `factores`, `score` y `cobertura_pct` a cada empresa del universo.

    Devuelve la lista ordenada por score descendente. Muta los dicts de entrada,
    igual que el resto de la pipeline.

    Tres decisiones que conviene conocer:

    - **Un factor se calcula con las métricas que haya.** Si a una empresa le
      faltan dos de las seis de calidad, su factor de calidad es la media de las
      cuatro disponibles, no una media con dos ceros dentro.
    - **Los pesos se renormalizan sobre los factores presentes.** Si a una
      empresa le falta el factor entero, su score se reparte entre los demás en
      vez de hundirse. Eso mantiene comparables a empresas con distinta cobertura
      de datos, pero la hace parecer más sólida de lo que es — por eso existe la
      tercera decisión.
    - **`cobertura_pct` viaja con el score.** Es el porcentaje de métricas
      posibles que la empresa tenía de verdad. Un score de 90 con cobertura del
      40% es una opinión sobre pocos datos, y quien lea la tabla tiene derecho a
      saberlo sin reconstruirlo.
    """
    if not universo:
        return []

    factores_usados = [f for f in pesos if f in FACTORES]

    # Rango percentil de cada métrica sobre TODO el universo. Aquí está la
    # diferencia con el screener actual: se rankea contra los 500, no contra los
    # 5 que sobrevivieron. Un nombre sube porque ha mejorado, no porque otro
    # se haya caído.
    percentiles: dict[str, list] = {}
    for nombre_factor in factores_usados:
        for metrica in FACTORES[nombre_factor]:
            valores = [e.get(metrica.clave) for e in universo]
            percentiles[metrica.clave] = rango_percentil(valores, metrica.mayor_es_mejor)

    total_metricas = sum(len(FACTORES[f]) for f in factores_usados)

    for i, empresa in enumerate(universo):
        puntuaciones_factor = {}
        metricas_presentes = 0

        for nombre_factor in factores_usados:
            disponibles = [
                percentiles[m.clave][i]
                for m in FACTORES[nombre_factor]
                if percentiles[m.clave][i] is not None
            ]
            metricas_presentes += len(disponibles)
            if disponibles:
                puntuaciones_factor[nombre_factor] = round(sum(disponibles) / len(disponibles), 1)

        empresa["factores"] = puntuaciones_factor

        peso_total = sum(pesos[f] for f in puntuaciones_factor)
        if peso_total > 0:
            empresa["score"] = round(
                sum(puntuaciones_factor[f] * pesos[f] for f in puntuaciones_factor) / peso_total,
                1,
            )
        else:
            empresa["score"] = None

        empresa["cobertura_pct"] = (
            round(metricas_presentes / total_metricas * 100) if total_metricas else 0
        )

    # Las empresas sin score van al final, no al principio: `None` no es un cero.
    universo.sort(key=lambda e: (e["score"] is not None, e.get("score") or 0), reverse=True)
    return universo


def describir_pesos(pesos: dict) -> str:
    """Una línea legible con la tesis del screener, para meterla en el JSON."""
    partes = [f"{f} {int(p)}%" for f, p in sorted(pesos.items(), key=lambda kv: -kv[1])]
    return " · ".join(partes)
