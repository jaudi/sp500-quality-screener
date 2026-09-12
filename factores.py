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
#
# Los cinco factores pesan en los tres screeners. Lo que cambia es la
# inclinación, nunca la ceguera: un screener de valor que no mire el crecimiento
# no distingue una empresa barata que crece de una barata que se encoge, y uno de
# crecimiento que no mire el precio no distingue un buen negocio de un buen
# negocio ya pagado. Poner ambos factores en los dos es, de hecho, lo que hace el
# PEG de Lynch —precio contra crecimiento— sin necesidad de un ratio aparte.
PESOS: dict[str, dict[str, float]] = {
    "sp500": {"value": 30, "quality": 25, "growth": 15, "momentum": 15, "expectativas": 15},
    "ibex35": {"value": 35, "quality": 25, "growth": 10, "momentum": 15, "expectativas": 15},
    "nasdaq100": {"growth": 30, "quality": 20, "value": 15, "momentum": 20, "expectativas": 15},
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


# Qué mide cada factor, en una frase. Es lo único de la metodología escrito a
# mano; todo lo demás se genera desde FACTORES y PESOS, para que la explicación
# publicada no pueda desviarse de la que se ejecutó.
GLOSA_FACTORES = {
    "value": "What you pay for what the business earns and owns. Built on normalised earnings rather than the last twelve months, so a company at the top of its cycle does not read as cheap.",
    "quality": "Whether the business earns its returns on real capital and turns profit into cash. Accruals and leverage sit here because both are ways a good-looking profit can fail to be one.",
    "growth": "How fast revenue and earnings are compounding, measured against a multi-year base rather than a single prior year — a comparison against one weak year is arithmetic, not growth.",
    "momentum": "Whether the price agrees. Deliberately the lowest weight in the value screens: a company that has fallen is cheaper, not worse, and momentum should not be able to veto it.",
    "expectativas": "How much growth today's price already demands, from the reverse DCF, against what the business has actually delivered. It is the only factor that scores the market's expectations rather than the company.",
}


def describir_metodologia(pesos: dict) -> dict:
    """La metodología completa, generada desde la configuración que se ejecuta.

    Se escribe en el JSON y el portal la renderiza tal cual. Que salga de
    `FACTORES` y `PESOS` y no de un texto paralelo es el punto: cambiar un peso
    cambia la explicación publicada en el mismo commit, sin que nadie tenga que
    acordarse de actualizarla.

    Un score compuesto es opaco por naturaleza — un 67 no se puede discutir. Por
    eso se publica también el desglose por factor de cada empresa: "barata y de
    calidad pero sin momentum" sí se puede discutir.
    """
    factores_usados = {f: p for f, p in pesos.items() if f in FACTORES}

    return {
        "approach": (
            "Factor scoring, not threshold filtering. Every company in the index is ranked against "
            "every other on each metric, and the ranks are combined with the weights below. Nothing "
            "is rejected for scoring poorly — only for being impossible to evaluate."
        ),
        "why_not_filters": (
            "A hard filter chain (ROE > 20% AND P/E < 20 AND ...) discards a company at 19.8% ROE as "
            "readily as one at 3%, empties the screen when no name clears every bar, throws away how "
            "far each company cleared or missed, and silently gives RSI the same weight as ROE. A "
            "cross-sectional rank has none of those problems."
        ),
        "why_ranks_not_values": (
            "Scores are built from percentile ranks rather than raw values. One company emerging from "
            "a cyclical trough at +1,368% earnings growth would otherwise dominate any average and "
            "turn a multi-factor score into a single column in disguise. A rank treats that as what it "
            "is — first place — and nothing more."
        ),
        "weights": {f: p for f, p in sorted(factores_usados.items(), key=lambda kv: -kv[1])},
        "weights_summary": describir_pesos(factores_usados),
        "factors": {
            nombre: {
                "weight_pct": peso,
                "what_it_measures": GLOSA_FACTORES.get(nombre, ""),
                "metrics": [
                    {
                        "name": m.clave,
                        "better_when": "higher" if m.mayor_es_mejor else "lower",
                    }
                    for m in FACTORES[nombre]
                ],
            }
            for nombre, peso in sorted(factores_usados.items(), key=lambda kv: -kv[1])
        },
        "missing_data": (
            "A metric a company does not report scores nothing — never zero. Scoring a gap as the "
            "worst possible value punishes a company for a data provider's omission, and that is the "
            "kind of error that never appears in the table and decides the order anyway. A factor is "
            "averaged over the metrics that exist, and the weights are renormalised over the factors "
            "that exist."
        ),
        "coverage": (
            "Because weights renormalise, a company with thin data scores as confidently as a complete "
            "one. cobertura_pct is published alongside every score for exactly that reason: 90 on 40% "
            "coverage is an opinion about very little."
        ),
        "what_the_score_is_not": (
            "The score is a position within this index, not a grade and not a valuation. A company's "
            "score moves when other companies move. It says nothing about whether the name is worth "
            "owning, and it is not a recommendation."
        ),
        "read_the_breakdown": (
            "The per-factor scores matter more than the total. 'Cheap and high quality but no momentum' "
            "is a statement you can argue with; a composite of 67 is not."
        ),
    }
