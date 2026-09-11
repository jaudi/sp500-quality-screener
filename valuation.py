"""
Etapa de valoración: DCF inverso ("¿qué está asumiendo el precio?") frente a
DCF histórico ("¿qué ha hecho de verdad la empresa?").

La idea, muy en la línea de Howard Marks, es no preguntarse si una empresa es
buena sino qué tiene que pasar para que el precio de hoy tenga sentido, y con
qué frecuencia ha pasado eso históricamente. Por eso aquí se calculan tres
cosas para cada empresa que supera el cribado:

  1. Crecimiento implícito: el crecimiento de FCF a 10 años que iguala el DCF
     a la capitalización actual. Es lo que el mercado está descontando.
  2. Crecimiento histórico y su DCF: el CAGR real del free cash flow en la
     ventana disponible, y el valor por acción que sale de proyectarlo.
  3. Probabilidad implícita: dada la distribución de crecimientos interanuales
     de la propia empresa, qué probabilidad tiene de alcanzar o superar el
     crecimiento que el precio exige.

Este módulo es solo aritmética y datos — no llama a Claude. El comentario
cualitativo lo escribe el agente de common.generar_informe_valoracion().
"""

import math

import pandas as pd
import yfinance as yf

# ==============================================================================
# PARÁMETROS DEL MODELO
# ==============================================================================
# Modelo en dos etapas: 10 años de crecimiento explícito + perpetuidad de Gordon.
HORIZONTE_ANIOS = 10

# Crecimiento a perpetuidad. 2.5% ≈ crecimiento nominal de largo plazo de una
# economía desarrollada: por encima de eso la empresa acabaría siendo más grande
# que el PIB, que es el error clásico que infla cualquier DCF.
CRECIMIENTO_TERMINAL = 0.025

# CAPM: ke = tasa libre de riesgo + beta × prima de riesgo.
PRIMA_RIESGO_MERCADO = 0.05

# La tasa libre de riesgo tiene que ir en la misma divisa que los flujos. Yahoo
# solo publica deuda del Tesoro de EE.UU. (^TNX/^IRX/^TYX): no hay Bund ni curva
# euro, así que para el resto de divisas se usa una constante documentada y el
# JSON dice cuál se aplicó y si vino en vivo o de aquí. Descontar una empresa que
# reporta en euros con el 10 años americano — lo que hacía la versión anterior —
# no era una aproximación, era mezclar dos economías con inflaciones distintas.
TASA_LIBRE_RIESGO_FALLBACK = {
    "USD": 0.040,
    "EUR": 0.027,   # Bund 10a aprox., revisar si se desvía mucho
    "GBP": 0.043,
    "CHF": 0.005,
}
TASA_LIBRE_RIESGO_POR_DEFECTO = 0.040

# La beta de Yahoo es ruidosa (ventanas cortas, tickers ilíquidos) y una beta de
# 0.1 o de 3.5 rompe el descuento. Se sigue acotando, pero la banda anterior
# ([0.5, 2.0] y ke en [7%, 15%]) era tan estrecha que recortaba betas legítimas:
# LOG.MC cotiza con 0.437 y NEM con 0.543, defensivas de verdad a las que se les
# estaba imponiendo un coste de capital superior al suyo. Ahora la banda solo
# atrapa valores imposibles, y cuando muerde se marca en beta_clamped/ke_clamped
# en lugar de desaparecer dentro del número.
BETA_MIN, BETA_MAX = 0.3, 2.5
KE_MIN, KE_MAX = 0.05, 0.18

# Por debajo de este R² la serie de FCF no se considera una tendencia y no se
# proyecta. Es el filtro que separa a un compounder de una cíclica que ha pasado
# por un suelo: con 4 puntos, un ajuste malo significa que el "crecimiento
# histórico" describe el recorrido de la serie, no el rumbo del negocio.
R2_MINIMO_PARA_PROYECTAR = 0.50

# Por encima de esta volatilidad logarítmica la probabilidad no se publica. El
# modelo ya sabía que por encima de ~1.0 no distingue nada; seguir enseñando un
# 52.3% con una etiqueta de "poco fiable" al lado es pedirle al lector que
# ignore la cifra más concreta de la fila, que no es como se leen las tablas.
LOG_STDEV_MAX_PUBLICABLE = 1.0

# Rango de búsqueda del crecimiento implícito. Fuera de aquí no se inventa un
# número: se reporta que el precio está fuera del rango que el modelo cubre.
# Es deliberadamente ancho: aquí no se está proyectando nada, se está
# despejando qué crecimiento hace cuadrar el precio, y esa incógnita puede caer
# legítimamente en valores extremos.
G_MIN_BUSQUEDA, G_MAX_BUSQUEDA = -0.30, 0.60

# Banda para el crecimiento que SÍ se proyecta (el DCF histórico). Es mucho más
# estrecha que la de búsqueda a propósito: el CAGR histórico de una cíclica
# puede ser absurdo como previsión. Newmont, saliendo de un suelo de FCF de 97M
# hasta 7.299M, arroja un CAGR del 88%; proyectarlo diez años daba un valor
# razonable de 8.335 $ contra un precio de 128 $ (+6.407%), que no es un
# hallazgo sino un artefacto del punto de partida. Se acota y se marca con
# historical_growth_capped para que el recorte sea visible, no silencioso.
G_MODELADO_MIN, G_MODELADO_MAX = -0.15, 0.25
TOLERANCIA_BISECCION = 1e-7
MAX_ITERACIONES = 200


# ==============================================================================
# DISTRIBUCIÓN t DE STUDENT (sin scipy)
# ==============================================================================
# yfinance da 4-5 estados anuales, o sea 3-4 crecimientos interanuales. Con esa
# n, estimar la desviación típica y tratarla como conocida (normal) subestima
# las colas justo donde importa. La t de Student con n-1 grados de libertad es
# la corrección correcta, y no merece la pena arrastrar scipy entero en el job
# de GitHub Actions solo por esta función.
def _fraccion_continua_beta(a: float, b: float, x: float) -> float:
    """Fracción continua de Lentz para la función beta incompleta."""
    minimo = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimo:
        d = minimo
    d = 1.0 / d
    h = d

    for m in range(1, MAX_ITERACIONES):
        m2 = 2 * m
        # Paso par
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < minimo:
            d = minimo
        c = 1.0 + num / c
        if abs(c) < minimo:
            c = minimo
        d = 1.0 / d
        h *= d * c
        # Paso impar
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < minimo:
            d = minimo
        c = 1.0 + num / c
        if abs(c) < minimo:
            c = minimo
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break

    return h


def beta_incompleta_regularizada(a: float, b: float, x: float) -> float:
    """I_x(a, b) — necesaria para la CDF de la t."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    ln_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(ln_factor)

    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _fraccion_continua_beta(a, b, x) / a
    # Fuera de la zona de convergencia rápida se usa la simetría
    # I_x(a,b) = 1 - I_{1-x}(b,a).
    return 1.0 - factor * _fraccion_continua_beta(b, a, 1.0 - x) / b


def cdf_t_student(x: float, grados_libertad: int) -> float:
    """P(T <= x) para una t de Student con `grados_libertad` gl."""
    if grados_libertad < 1:
        raise ValueError("La t de Student necesita al menos 1 grado de libertad.")

    z = grados_libertad / (grados_libertad + x * x)
    cola = 0.5 * beta_incompleta_regularizada(grados_libertad / 2.0, 0.5, z)
    return 1.0 - cola if x > 0 else cola


# ==============================================================================
# LECTURA DE ESTADOS FINANCIEROS
# ==============================================================================
def _serie_anual(estado, fila: str) -> list[float]:
    """Extrae una fila de un estado de yfinance de más antiguo a más reciente.

    yfinance devuelve las columnas de más reciente a más antigua y mete NaN en
    los ejercicios que no tiene, así que hay que invertir y limpiar.
    """
    if estado is None or estado.empty or fila not in estado.index:
        return []

    valores = []
    for valor in reversed(list(estado.loc[fila].values)):
        if valor is None or pd.isna(valor):
            continue
        valores.append(float(valor))
    return valores


def serie_free_cash_flow(ticker_obj) -> tuple[list[float], str]:
    """Devuelve (serie de FCF anual, etiqueta de la fuente usada).

    Se prefiere la fila 'Free Cash Flow' que ya publica yfinance; si no está, se
    reconstruye como flujo de explotación menos capex (capex viene en negativo,
    por eso se suma).
    """
    estado = ticker_obj.cashflow

    directo = _serie_anual(estado, "Free Cash Flow")
    if len(directo) >= 2:
        return directo, "Free Cash Flow (reported)"

    cfo = _serie_anual(estado, "Operating Cash Flow")
    capex = _serie_anual(estado, "Capital Expenditure")
    if len(cfo) >= 2 and len(cfo) == len(capex):
        return [o + c for o, c in zip(cfo, capex)], "Operating Cash Flow - capex"

    return [], "unavailable"


def cagr(serie: list[float]) -> float | None:
    """CAGR punta a punta. None si no hay dos puntos con ambos extremos positivos.

    Con un extremo negativo el CAGR no está definido (una raíz de un número
    negativo, o un crecimiento "infinito" al salir de pérdidas), y devolver un
    número igualmente es la forma más rápida de publicar una valoración absurda.
    """
    if len(serie) < 2:
        return None
    inicio, fin = serie[0], serie[-1]
    if inicio <= 0 or fin <= 0:
        return None

    anios = len(serie) - 1
    return (fin / inicio) ** (1.0 / anios) - 1.0


def tendencia_log_lineal(serie: list[float]) -> tuple[float | None, float | None]:
    """Ajusta una recta por mínimos cuadrados sobre log(FCF). Devuelve (crecimiento, R²).

    El CAGR solo mira los dos extremos, así que hereda todo lo que tengan de
    raro: el 88,54% de Newmont sale de comparar 1.089 con 7.299 e ignora por
    completo que por el medio hubo un suelo de 97. La recta usa todos los puntos,
    y sobre todo el R² dice si la serie se parece a una tendencia o es un
    recorrido accidentado — que con cuatro observaciones es la pregunta que de
    verdad importa antes de proyectar nada.
    """
    puntos = [(i, math.log(v)) for i, v in enumerate(serie) if v > 0]
    if len(puntos) < 3:
        # Con dos puntos la recta pasa exacta por ambos y el R² es 1 por
        # construcción, que sería un aprobado automático sin información.
        return None, None

    n = len(puntos)
    media_x = sum(x for x, _ in puntos) / n
    media_y = sum(y for _, y in puntos) / n

    sxy = sum((x - media_x) * (y - media_y) for x, y in puntos)
    sxx = sum((x - media_x) ** 2 for x, _ in puntos)
    if sxx == 0:
        return None, None

    pendiente = sxy / sxx
    syy = sum((y - media_y) ** 2 for _, y in puntos)
    r2 = 1.0 if syy == 0 else max(0.0, min(1.0, (sxy * sxy) / (sxx * syy)))

    return math.exp(pendiente) - 1.0, r2


def log_crecimientos_interanuales(serie: list[float]) -> list[float]:
    """Crecimientos año contra año en logaritmos, saltando tramos no positivos.

    En aritmético una caída y su recuperación no se cancelan: la serie de FCF de
    Newmont [1.089, 97, 2.961, 7.299] da -91%, +2.953% y +147%, o sea una media
    de +1.003% anual con una desviación de 1.693%. Eso no describe el negocio,
    describe el punto por el que pasó la serie, y cualquier probabilidad
    construida encima es ruido con formato de estadística.

    En logaritmos ese viaje de ida y vuelta sí se cancela (ln(0.09) = -2.4 frente
    a ln(30.5) = +3.4), que es el tratamiento económicamente correcto y además el
    supuesto habitual — crecimiento lognormal. El precio a pagar es descartar los
    tramos que tocan FCF negativo, donde el logaritmo no existe.
    """
    salidas = []
    for anterior, actual in zip(serie, serie[1:]):
        if anterior <= 0 or actual <= 0:
            continue
        salidas.append(math.log(actual / anterior))
    return salidas


# ==============================================================================
# COSTE DE CAPITAL Y DCF
# ==============================================================================
def obtener_tasa_libre_riesgo(divisa: str | None = "USD") -> tuple[float, str]:
    """Tasa libre de riesgo en la divisa de los flujos. Devuelve (tasa, fuente).

    Solo el dólar tiene cotización en vivo — Yahoo publica ^TNX pero ningún
    equivalente europeo. Para el resto se devuelve la constante documentada y se
    dice que lo es, para que quede claro en el JSON qué número es de mercado y
    cuál es una hipótesis que hay que revisar de vez en cuando.
    """
    divisa = (divisa or "USD").upper()

    if divisa == "USD":
        try:
            hist = yf.Ticker("^TNX").history(period="5d")
            if not hist.empty:
                tasa = float(hist["Close"].iloc[-1]) / 100.0
                if 0.0 < tasa < 0.15:
                    return tasa, "live (^TNX 10y Treasury)"
        except Exception:
            pass

    tasa = TASA_LIBRE_RIESGO_FALLBACK.get(divisa, TASA_LIBRE_RIESGO_POR_DEFECTO)
    return tasa, f"static default for {divisa}"


def coste_capital_propio(beta, tasa_libre_riesgo: float) -> dict:
    """CAPM acotado, informando de si el recorte llegó a morder."""
    try:
        beta_valor = float(beta)
    except (TypeError, ValueError):
        beta_valor = None
    if beta_valor is None or math.isnan(beta_valor):
        beta_valor = 1.0
        beta_ausente = True
    else:
        beta_ausente = False

    beta_acotada = min(max(beta_valor, BETA_MIN), BETA_MAX)
    ke_bruto = tasa_libre_riesgo + beta_acotada * PRIMA_RIESGO_MERCADO
    ke = min(max(ke_bruto, KE_MIN), KE_MAX)

    return {
        "ke": ke,
        "beta_used": round(beta_acotada, 3),
        "beta_reported": None if beta_ausente else round(beta_valor, 3),
        "beta_clamped": beta_acotada != beta_valor,
        "ke_clamped": ke != ke_bruto,
        "beta_missing": beta_ausente,
    }


def peso_valor_terminal(fcf_base: float, crecimiento: float, ke: float) -> float:
    """Fracción del DCF que sale de la perpetuidad, no de los 10 años explícitos.

    Es la cifra que dice cuánto pesa de verdad la hipótesis de crecimiento
    terminal. Si el 85% del valor está en el terminal, discutir el crecimiento
    de los años 1 a 10 es discutir el 15% del resultado, y el 2,5% fijo — igual
    para una minera de oro que para una retailer — es quien manda.
    """
    explicito = sum(
        fcf_base * (1.0 + crecimiento) ** anio / (1.0 + ke) ** anio
        for anio in range(1, HORIZONTE_ANIOS + 1)
    )
    fcf_final = fcf_base * (1.0 + crecimiento) ** HORIZONTE_ANIOS
    terminal = (
        fcf_final * (1.0 + CRECIMIENTO_TERMINAL) / (ke - CRECIMIENTO_TERMINAL)
    ) / (1.0 + ke) ** HORIZONTE_ANIOS

    total = explicito + terminal
    return terminal / total if total else 0.0


def valor_equity(fcf_base: float, crecimiento: float, ke: float) -> float:
    """DCF en dos etapas sobre free cash flow apalancado.

    Se descuenta el FCF (ya post-intereses) al coste de capital propio y el
    resultado se compara directamente contra la capitalización bursátil, sin
    puente de deuda neta. Es una simplificación consciente: el FCFF con WACC
    sería más ortodoxo, pero exige campos que yfinance deja vacíos en bastantes
    tickers, y el objetivo aquí es un modelo que corra sobre 500 empresas sin
    caerse, no un modelo de banca de inversión.
    """
    valor_actual = 0.0
    fcf = fcf_base

    for anio in range(1, HORIZONTE_ANIOS + 1):
        fcf = fcf_base * (1.0 + crecimiento) ** anio
        valor_actual += fcf / (1.0 + ke) ** anio

    # Perpetuidad de Gordon sobre el FCF del último año explícito.
    valor_terminal = fcf * (1.0 + CRECIMIENTO_TERMINAL) / (ke - CRECIMIENTO_TERMINAL)
    valor_actual += valor_terminal / (1.0 + ke) ** HORIZONTE_ANIOS

    return valor_actual


def crecimiento_implicito(capitalizacion: float, fcf_base: float, ke: float) -> tuple[float | None, str]:
    """DCF inverso: resuelve el crecimiento que iguala el DCF a la capitalización.

    valor_equity es monótona creciente en `crecimiento`, así que la bisección es
    segura. Devuelve (crecimiento, estado); si el precio queda fuera del rango
    de búsqueda se dice explícitamente en vez de devolver el extremo como si
    fuera una solución.
    """
    valor_min = valor_equity(fcf_base, G_MIN_BUSQUEDA, ke)
    valor_max = valor_equity(fcf_base, G_MAX_BUSQUEDA, ke)

    if capitalizacion <= valor_min:
        return None, "below_model_range"
    if capitalizacion >= valor_max:
        return None, "above_model_range"

    bajo, alto = G_MIN_BUSQUEDA, G_MAX_BUSQUEDA
    for _ in range(MAX_ITERACIONES):
        medio = (bajo + alto) / 2.0
        valor = valor_equity(fcf_base, medio, ke)
        if abs(valor - capitalizacion) / capitalizacion < TOLERANCIA_BISECCION:
            return medio, "ok"
        if valor < capitalizacion:
            bajo = medio
        else:
            alto = medio

    return (bajo + alto) / 2.0, "ok"


# ==============================================================================
# PROBABILIDAD IMPLÍCITA
# ==============================================================================
def probabilidad_de_alcanzar(objetivo: float, log_historicos: list[float]) -> dict:
    """P(crecimiento >= objetivo) según la propia historia de la empresa.

    `log_historicos` son crecimientos interanuales en logaritmos (ver
    log_crecimientos_interanuales). Se ajusta una t de Student sobre ellos y se
    compara contra el objetivo llevado también a logaritmos.

    Con n pequeña esto NO es una probabilidad de mercado: es "con qué frecuencia
    esta empresa ha crecido así", que es exactamente la pregunta de segundo
    nivel. Se devuelve n para que el informe pueda decir sobre cuántas
    observaciones se apoya.
    """
    n = len(log_historicos)
    # La media se devuelve destransformada, o sea geométrica, para que sea
    # comparable con el CAGR y con el crecimiento implícito. La media de los
    # logaritmos no significa nada para quien lea el informe.
    media_log = sum(log_historicos) / n if n else None
    base = {
        "observations": n,
        "historical_mean_growth_pct": round((math.exp(media_log) - 1) * 100, 2) if n else None,
        "basis": "log growth",
    }

    if n < 2:
        return {
            **base,
            "probability_pct": None,
            "reason": "fewer than 2 usable year-on-year growth observations",
        }

    if objetivo <= -1.0:
        return {**base, "probability_pct": None, "reason": "implied growth at or below -100%"}

    objetivo_log = math.log(1.0 + objetivo)
    media = media_log
    varianza = sum((g - media) ** 2 for g in log_historicos) / (n - 1)
    desviacion = math.sqrt(varianza)
    base["historical_log_stdev"] = round(desviacion, 3)

    # Comparar contra cero exacto no sirve: una serie plana como [0.10, 0.10,
    # 0.10] deja una varianza residual de ~1e-35 por redondeo binario, y dividir
    # por esa desviación produce un t-stat con pinta de válido a partir de puro
    # ruido de coma flotante. Se trata como degenerada cualquier desviación
    # despreciable frente a la propia media.
    if desviacion <= max(1e-9, abs(media) * 1e-9):
        # Serie perfectamente plana: la probabilidad degenera a 0 o 1.
        return {
            **base,
            # El margen es por el empate exacto: una serie que crece un 10%
            # limpio y un objetivo del 10% deberían dar 100% (la probabilidad es
            # P(>=)), pero log(1.10) y la media de los logaritmos difieren en el
            # último bit y sin margen el resultado saltaría entre 0 y 100.
            "probability_pct": 100.0 if objetivo_log <= media + 1e-12 else 0.0,
            "reason": "historical growth has zero variance",
        }

    # Por encima del umbral la t está tan aplanada que la probabilidad no
    # distingue un 30% de un 70%. Se retiene en lugar de publicarla con una
    # advertencia al lado: en una tabla, el número gana siempre a la nota.
    if desviacion > LOG_STDEV_MAX_PUBLICABLE:
        return {
            **base,
            "probability_pct": None,
            "reason": (
                f"cash flow too volatile to estimate (log stdev {desviacion:.2f} "
                f"> {LOG_STDEV_MAX_PUBLICABLE}); any probability here would be noise"
            ),
        }

    t_stat = (objetivo_log - media) / desviacion
    probabilidad = 1.0 - cdf_t_student(t_stat, n - 1)

    return {
        **base,
        "t_statistic": round(t_stat, 3),
        "degrees_of_freedom": n - 1,
        "probability_pct": round(probabilidad * 100, 1),
        # Con n=3 el intervalo es enorme y esconderlo daba una falsa precisión.
        # Se acota reajustando la desviación por su propia incertidumbre (chi2
        # aproximada vía el factor 1 ± 1/sqrt(2(n-1))), que basta para enseñar
        # el orden de magnitud del error sin fingir un método exacto.
        "probability_range_pct": _rango_probabilidad(objetivo_log, media, desviacion, n),
    }


def _rango_probabilidad(objetivo_log: float, media: float, desviacion: float, n: int) -> list:
    """Banda aproximada de la probabilidad, dada la incertidumbre de la propia sigma."""
    factor = 1.0 / math.sqrt(2.0 * (n - 1))
    extremos = []
    for ajuste in (1.0 - factor, 1.0 + factor):
        sigma = desviacion * ajuste
        if sigma <= 0:
            continue
        p = 1.0 - cdf_t_student((objetivo_log - media) / sigma, n - 1)
        extremos.append(round(p * 100, 1))

    if not extremos:
        return None
    return [min(extremos), max(extremos)]


# ==============================================================================
# ORQUESTACIÓN POR EMPRESA
# ==============================================================================
def analizar_valoracion(ticker: str, precio_cribado: float | None = None) -> dict:
    """Calcula DCF inverso + DCF histórico + probabilidad implícita para un ticker.

    Lanza ValueError con un motivo legible cuando faltan los datos mínimos, para
    que el llamante lo registre como fallo en vez de publicar una valoración
    construida sobre huecos.
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info or {}

    capitalizacion = info.get("marketCap")
    if not capitalizacion or capitalizacion <= 0:
        raise ValueError("no market cap available")

    # Si la divisa de cotización y la de los estados no coinciden, la
    # capitalización y el FCF no son comparables y el DCF saldría desplazado por
    # el tipo de cambio, no por el negocio.
    divisa_precio = info.get("currency")
    divisa_estados = info.get("financialCurrency")
    if divisa_precio and divisa_estados and divisa_precio != divisa_estados:
        raise ValueError(f"currency mismatch: quoted in {divisa_precio}, reports in {divisa_estados}")

    serie_fcf, fuente_fcf = serie_free_cash_flow(ticker_obj)
    if len(serie_fcf) < 2:
        raise ValueError("fewer than 2 years of free cash flow history")

    fcf_base = serie_fcf[-1]
    if fcf_base <= 0:
        raise ValueError("latest free cash flow is negative — reverse DCF is not meaningful")

    # La tasa se resuelve por empresa, no una vez para toda la tanda: el IBEX
    # reporta en euros y el S&P en dólares, y son curvas distintas.
    tasa_libre_riesgo, fuente_tasa = obtener_tasa_libre_riesgo(divisa_estados or divisa_precio)

    capm = coste_capital_propio(info.get("beta"), tasa_libre_riesgo)
    ke = capm["ke"]

    # 1. Lo que el precio está asumiendo.
    g_implicito, estado_implicito = crecimiento_implicito(capitalizacion, fcf_base, ke)

    # 2. Lo que la empresa ha hecho de verdad. Se calculan las dos lecturas: el
    #    CAGR punta a punta, que es lo que pediría cualquiera, y la recta sobre
    #    los logaritmos, que usa todos los puntos y viene con un R² que dice si
    #    la serie merece llamarse tendencia.
    g_historico = cagr(serie_fcf)
    g_tendencia, r2 = tendencia_log_lineal(serie_fcf)
    historicos = log_crecimientos_interanuales(serie_fcf)

    # Corroboración con ingresos: si el FCF se dispara pero la facturación no se
    # mueve, el movimiento viene de circulante, de un capex aplazado o de algo
    # no recurrente, no de que el negocio esté creciendo. No prueba cuál es,
    # pero marca la diferencia para que no se lea como tendencia.
    serie_ingresos = _serie_anual(ticker_obj.financials, "Total Revenue")
    g_ingresos = cagr(serie_ingresos)

    acciones = info.get("sharesOutstanding")
    precio = precio_cribado or info.get("currentPrice")

    valor_por_accion = None
    potencial_pct = None
    g_modelado = None
    crecimiento_acotado = False
    peso_terminal = None
    motivo_sin_dcf = None

    # La proyección solo sale si la serie se comporta como una tendencia. Antes
    # se proyectaba siempre y el recorte a la banda se anunciaba en una nota,
    # pero eso publicaba un valor por acción que era función del límite elegido
    # y no de la empresa: Newmont daba +520% porque el modelo había chocado con
    # su propio techo del 25%, no porque valiera eso.
    if g_tendencia is None or r2 is None:
        motivo_sin_dcf = "not enough usable free cash flow points to fit a trend"
    elif r2 < R2_MINIMO_PARA_PROYECTAR:
        motivo_sin_dcf = (
            f"free cash flow does not follow a trend (R2 {r2:.2f} < {R2_MINIMO_PARA_PROYECTAR}); "
            "projecting it would describe the path of the series, not the business"
        )
    else:
        g_modelado = min(max(g_tendencia, G_MODELADO_MIN), G_MODELADO_MAX)
        crecimiento_acotado = g_modelado != g_tendencia
        if crecimiento_acotado:
            motivo_sin_dcf = (
                f"trend growth {g_tendencia * 100:.1f}% sits outside the projectable band "
                f"[{G_MODELADO_MIN * 100:.0f}%, {G_MODELADO_MAX * 100:.0f}%]; any value would be "
                "a function of that boundary rather than of the company"
            )
        elif acciones:
            valor_historico = valor_equity(fcf_base, g_modelado, ke)
            peso_terminal = round(peso_valor_terminal(fcf_base, g_modelado, ke) * 100, 1)
            valor_por_accion = valor_historico / acciones
            if precio:
                potencial_pct = round((valor_por_accion / precio - 1.0) * 100, 1)

    # 3. Qué probabilidad asigna la historia a lo que el precio exige.
    probabilidad = (
        probabilidad_de_alcanzar(g_implicito, historicos)
        if g_implicito is not None
        else {"observations": len(historicos), "probability_pct": None,
              "reason": f"implied growth {estado_implicito}"}
    )

    # La brecha se mide contra la tendencia, no contra el CAGR: comparar lo que
    # el precio exige con un número anclado en dos extremos daba lecturas como
    # los -87pp de Newmont, que medían el suelo de 2022 más que al negocio.
    #
    # Y solo se publica si esa tendencia ha pasado el test de R². Restar de un
    # ajuste que el propio modelo ha descartado devuelve aritmética, no una
    # señal: Newmont daba -147,6pp contra una "tendencia" del 149% que ya se
    # había declarado no proyectable dos líneas más arriba.
    brecha_pp = None
    tendencia_fiable = r2 is not None and r2 >= R2_MINIMO_PARA_PROYECTAR
    if g_implicito is not None and tendencia_fiable and g_tendencia is not None:
        brecha_pp = round((g_implicito - g_tendencia) * 100, 1)

    # Divergencia FCF vs ingresos, en puntos porcentuales.
    divergencia_pp = None
    if g_tendencia is not None and g_ingresos is not None:
        divergencia_pp = round((g_tendencia - g_ingresos) * 100, 1)

    return {
        "ticker": ticker,
        "market_cap": capitalizacion,
        "currency": divisa_estados or divisa_precio,
        "price": round(precio, 2) if precio else None,
        "fcf_latest": fcf_base,
        "fcf_source": fuente_fcf,
        "fcf_years": len(serie_fcf),
        "fcf_series": [round(v, 0) for v in serie_fcf],
        "cost_of_equity_pct": round(ke * 100, 2),
        "beta_used": capm["beta_used"],
        "beta_reported": capm["beta_reported"],
        "beta_clamped": capm["beta_clamped"],
        "ke_clamped": capm["ke_clamped"],
        "beta_missing": capm["beta_missing"],
        "risk_free_rate_pct": round(tasa_libre_riesgo * 100, 2),
        "risk_free_source": fuente_tasa,
        "implied_growth_pct": round(g_implicito * 100, 2) if g_implicito is not None else None,
        "implied_growth_status": estado_implicito,
        "historical_growth_pct": round(g_historico * 100, 2) if g_historico is not None else None,
        "trend_growth_pct": round(g_tendencia * 100, 2) if g_tendencia is not None else None,
        "trend_r2": round(r2, 3) if r2 is not None else None,
        "revenue_growth_pct": round(g_ingresos * 100, 2) if g_ingresos is not None else None,
        "fcf_vs_revenue_divergence_pp": divergencia_pp,
        "modelled_growth_pct": round(g_modelado * 100, 2) if g_modelado is not None else None,
        "historical_growth_capped": crecimiento_acotado,
        "gap_pp": brecha_pp,
        "dcf_value_per_share": round(valor_por_accion, 2) if valor_por_accion else None,
        "dcf_upside_pct": potencial_pct,
        "dcf_terminal_value_share_pct": peso_terminal,
        "dcf_skipped_reason": motivo_sin_dcf,
        "probability": probabilidad,
    }


def analizar_valoraciones(
    empresas: list,
    pausa_entre_tickers: float = 0.4,
) -> tuple[list, list]:
    """Ejecuta analizar_valoracion sobre las empresas que superaron el cribado.

    Devuelve (valoraciones, fallidos). Un ticker que falla no tumba la etapa:
    el cribado ya tiene valor por sí solo y el resto de valoraciones también.
    """
    import time

    print("\n" + "=" * 70)
    print("🧮 Etapa de valoración (DCF inverso)")
    print("=" * 70)

    valoraciones, fallidos = [], []
    for empresa in empresas:
        ticker = empresa["ticker"]
        try:
            valoracion = analizar_valoracion(
                ticker,
                precio_cribado=empresa.get("precio_actual"),
            )
            valoracion["nombre"] = empresa.get("nombre")
            valoracion["sector"] = empresa.get("sector")
            valoraciones.append(valoracion)

            implicito = valoracion["implied_growth_pct"]
            tendencia = valoracion["trend_growth_pct"]
            r2 = valoracion["trend_r2"]
            prob = valoracion["probability"].get("probability_pct")
            prob_txt = f"{prob}%" if prob is not None else "n/d (demasiado volátil)"
            dcf_txt = (
                f"DCF {valoracion['dcf_value_per_share']}"
                if valoracion["dcf_value_per_share"] is not None
                else "sin DCF"
            )
            print(
                f"   {ticker}: implícito {implicito}% vs tendencia {tendencia}% "
                f"(R2 {r2}) → prob {prob_txt} · {dcf_txt}"
            )
        except Exception as e:
            motivo = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            fallidos.append({"ticker": ticker, "error": motivo})
            print(f"   ⚠️  {ticker}: sin valoración — {motivo}")
        finally:
            time.sleep(pausa_entre_tickers)

    return valoraciones, fallidos
