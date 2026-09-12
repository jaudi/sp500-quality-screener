# Roadmap de los screeners

Trabajo acordado, ordenado por relación esfuerzo/valor. Cada punto lleva la
dificultad comprobada contra lo que yfinance devuelve de verdad, no contra lo que
debería devolver.

---

## La dirección

**Base intelectual: la escuela americana.** Howard Marks (ciclo, qué está ya
descontado, riesgo como pérdida permanente), Peter Lynch (el múltiplo no
significa nada sin el crecimiento, las cíclicas engañan), Jim Simons (enséñame
la evidencia, ranking transversal, nada de umbrales inventados).

**Dos preguntas, dos diseños:**

- **Sectores tradicionales** (S&P 500, IBEX 35) → **value + calidad**
- **Nasdaq-100** → **crecimiento + momentum**

**Y una sola arquitectura para los dos: multifactor.**

### Por qué multifactor en vez de filtros binarios

Hoy los screeners son cadenas de filtros duros: ROE > 20% Y P/E < 20 Y RSI > 30…
Una empresa con ROE del 19,8% desaparece igual que una con ROE del 3%. Eso tiene
cuatro problemas que el enfoque por factores resuelve de golpe:

1. **Los umbrales son inventados.** 20%, 20x, 30 de RSI. Ninguno está validado, y
   el de crecimiento del Nasdaq se calibró para que pasaran ~11 nombres, que es
   ajustar a un resultado estético. Un ranking transversal no necesita umbral.
2. **La pantalla se vacía.** Con 35 valores y 5 filtros, el IBEX devuelve 3
   nombres. Un ranking siempre devuelve un top N.
3. **Se pierde toda la información del margen.** Un filtro binario tira el dato
   de *por cuánto* pasó o falló cada empresa.
4. **No hay forma de ponderar.** Hoy el RSI pesa lo mismo que el ROE. Con
   factores se decide explícitamente, y se puede cambiar y medir.

### El diseño

Cinco factores, cada uno un rango percentil 0-100 **dentro del universo
completo** (no dentro de los supervivientes):

| Factor | Compuesto por |
|---|---|
| **Value** | P/E normalizado, P/B, FCF yield, EV/EBIT |
| **Quality** | ROIC, márgenes, devengos, deuda neta/EBITDA, cobertura de intereses |
| **Growth** | crecimiento normalizado de ingresos y beneficios |
| **Momentum** | retorno 6m y 12m, estructura MA50/MA200 |
| **Expectativas** | crecimiento implícito del DCF inverso vs. tendencia real |

Y una ponderación distinta por screener, que es donde vive la tesis de cada uno:

```
S&P 500     Value 35 · Quality 35 · Momentum 15 · Expectativas 15
IBEX 35     Value 40 · Quality 35 · Momentum 10 · Expectativas 15
Nasdaq-100  Growth 40 · Quality 20 · Momentum 25 · Expectativas 15
```

Los tres publican el score total **y el desglose por factor**, que es lo que
convierte la tabla en algo que se puede discutir: no "pasó el filtro", sino
"barata y de calidad pero sin momentum", que es una frase con contenido.

El factor **Expectativas** es la aportación propia y la que Marks reconocería:
ya existe el DCF inverso, solo hay que convertirlo de informe final en una
columna que puntúe. Penaliza a las empresas cuyo precio exige mucho más de lo
que el negocio ha entregado nunca.

**Lo que sigue siendo un filtro duro, no un factor:** liquidez mínima, FCF
positivo donde la valoración lo necesita, y poco más. Un filtro debe eliminar lo
que no se puede analizar, no lo que puntúa bajo.

---

## Hecho

- [x] **Medidor de rendimiento forward** (`medir.py` + `performance.yml`).
      Reconstruye precio y fecha de entrada desde el historial de git —ya estaban
      recogidos sin que nadie lo planease— y compara contra el índice.
- [x] **Métricas en el glosario.** Los 12 conceptos de las tablas viven en
      `/glossary` y las cabeceras de columna enlazan a su definición.
- [x] **Ranking transversal** — existe en el Nasdaq (`_puntuar_y_ordenar`). Es el
      germen del motor multifactor: generalizarlo es el punto 1 de abajo.

## Prioridad 1 — el motor multifactor

- [x] **Pasada de recogida** (`recolectar_universo` en `common.py`). Recoge las
      métricas de todo el universo sin descartar por puntuación. Medido: ~1 s por
      ticker incluida la llamada a las cuentas anuales, así que el S&P 500 entero
      son ~8 min de Actions — gratis en repo público, y sin coste de API porque
      yfinance no cobra. Incluye beneficio normalizado, devengos de Sloan,
      cobertura de intereses, deuda neta/EBITDA, margen operativo y P/B.
- [x] **Runner multifactor** (`run_pipeline_multifactor`), con las dos etapas:
      ranking del universo con los 4 factores baratos → lista corta → DCF inverso
      sobre esos → `expectativas` y score final. Los 4 primeros conservan su
      percentil contra el índice entero; sólo `expectativas` se rankea dentro de
      la lista corta, porque fuera de ella no existe. `top_n_informe` fija cuántas
      empresas entran al prompt de Claude, que es lo único facturado.
- [x] **ROIC y EV/EBIT.** Estaban declarados en `FACTORES` desde el primer commit
      sin que nadie los calculara. ROIC sale de NOPAT sobre `Invested Capital`
      del balance; `returnOnCapital` de yfinance viene vacío en todos los tickers.
- [x] **`test_recoleccion.py`** — comprueba que toda métrica declarada en
      `FACTORES` se recoge de verdad. El fallo de "métrica declarada que nunca se
      calcula" había aparecido ya tres veces (devengos, ROIC, EV/EBIT) y no da
      error: el factor deja de pesar en silencio y el score publicado no es el que
      dice la metodología.
- [ ] **Cambiar los tres entry points a `run_pipeline_multifactor`.** El motor
      está probado de punta a punta pero ningún screener lo usa todavía: los tres
      siguen llamando a `filtrar_acciones_*`. Es el paso que lo pone en
      producción. *Dificultad: baja.*
- [ ] **Decidir qué hacer con `expectativas` ausente.** El test de R² es estricto
      a propósito, así que en la prueba sólo 4 de 8 empresas obtuvieron el factor.
      Quien no lo tiene no es penalizado —sus otros factores se reponderan—
      mientras que quien lo tiene malo sí. Asimetría real, sin decidir.
- [ ] **Usar `beneficio_en_pico`.** Ya se calcula y no se usa. NEM sigue saliendo
      primera en la prueba pese a que su P/E normalizado es 72x, porque calidad y
      momentum la sostienen. Decidir si el pico es penalización dentro de value o
      una marca visible en la tabla. *Dificultad: baja, decisión de criterio.*
- [x] **Motor de factores** (`factores.py` + `test_factores.py`). Rango
      percentil por métrica, media por factor, combinación ponderada. Aritmética
      pura: se prueba entero sin red y sin tokens, como `valuation.py`. Los pesos
      de los tres screeners viven en `PESOS` y las 33 pruebas verifican que
      cambiarlos cambia el orden, que un outlier no distorsiona la escala y que
      un hueco de datos no se convierte en un cero.
- [ ] **Rankear sobre el universo completo, no sobre los supervivientes.** Hoy el
      score del Nasdaq se calcula entre los 11 que pasan, así que un nombre sube
      porque otro salió, no porque haya mejorado. *Dificultad: baja una vez
      hechas las dos pasadas.*
- [ ] **Beneficio normalizado (media de 5 años) en todos los ratios.** Es lo que
      arregla de raíz el problema de CF y NEM, y el denominador correcto del PEG
      —sin normalizar, el PEG *empeora* el sesgo cíclico en vez de arreglarlo.
      `financials` da 5 años. *Dificultad: media.*
- [ ] **Flag de pico cíclico.** Beneficio actual en máximo de 5 años. En una
      cíclica eso es señal de venta, no de compra (Lynch). *Dificultad: media.*
- [ ] **Expectativas como factor.** Convertir el crecimiento implícito del DCF
      inverso en una columna puntuada. *Dificultad: baja — el dato ya se calcula.*

## Prioridad 2 — lo barato

El dato ya está en `info`; el bloque entero cabe en una tarde y alimenta
directamente los factores Value y Quality.

- [ ] P/B (`priceToBook`)
- [ ] FCF yield
- [ ] Payout sobre FCF, no sobre beneficio
- [ ] Conversión beneficio → FCF
- [ ] Ratio de devengos (accruals) — anomalía académicamente documentada, la
      única métrica de la lista que Simons defendería sin discutir
- [ ] PEG con denominador normalizado
- [ ] Regla del 40 para el software del Nasdaq

## Prioridad 3 — calidad y estructura

- [ ] **ROIC en vez de ROE.** `returnOnCapital` no existe en yfinance para ningún
      ticker; hay que calcularlo con balance + cuenta de resultados.
      *Dificultad: media-alta.*
- [ ] **Deuda neta/EBITDA y cobertura de intereses** en lugar de D/E. Un ratio
      estático no distingue deuda barata a 2032 de un vencimiento el año que
      viene (Marks). *Dificultad: media.*
- [ ] **Percentiles relativos al sector.** ROE > 20% significa cosas distintas en
      un banco y en software. Con dos pasadas ya hechas, esto es rankear dentro
      de cada sector en vez de dentro del universo. *Dificultad: baja tras la P1.*
- [ ] **Umbrales relativos a la propia historia.** "P/E en el decil bajo de su
      rango de 10 años" en vez de "P/E < 20". Requiere cruzar precios con BPA
      histórico alineado en tiempo. *Dificultad: alta.*
- [ ] **Regla de venta.** Los screeners dicen qué comprar y nunca qué hacer
      después; el medidor asume buy-and-hold por omisión. *Dificultad: baja en
      código, alta en criterio.*
- [ ] **Ampliar el IBEX a universo ibérico** (añadir el PSI): de 35 a ~120
      nombres. Con 35 nunca habrá señal. Ojo: más bancos, más huecos — `SAN.MC`
      no trae `freeCashflow` ni `ebitda` en yfinance.

## Descartado — y por qué

- **Backtest histórico de verdad.** El único que daría evidencia real y el único
  que no es viable aquí. yfinance devuelve fundamentales actuales, no los
  conocidos en cada fecha: cualquier backtest tendría sesgo de anticipación y
  diría que todo funciona maravillosamente. Peor que no tenerlo. Requiere datos
  point-in-time de pago (Sharadar, Compustat) — decisión de presupuesto.
  Mientras tanto, `medir.py` acumula evidencia forward, que es lenta pero limpia.
- **Muro de vencimientos de deuda.** No está en yfinance. Habría que parsear
  filings de la SEC y no cubriría el IBEX.
- **Compras de insiders.** El dato existe pero llega demasiado irregular para un
  job desatendido.

---

## Nota sobre el momentum

En los screeners de sectores tradicionales el momentum baja de filtro duro a
**factor con peso bajo** (10-15%). No se elimina: la evidencia de que el momentum
funciona es de las más sólidas que hay, y tirarla por gusto estético sería el
tipo de decisión que Simons no perdonaría.

Pero deja de *descartar* empresas. Hoy una compañía excelente y barata desaparece
del cribado por cotizar por debajo de su media de 50 días — es decir, justo
cuando más barata está. Eso es indefendible en un screener de value, y es lo
primero que miraría un inversor de esa escuela antes de decidir si sigue leyendo.

En el Nasdaq el momentum mantiene peso alto (25%), porque ahí el screener dice
explícitamente que es de crecimiento y momentum y no pretende otra cosa.
