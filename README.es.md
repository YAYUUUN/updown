# updown: ¿subirá?, con su historial honesto al lado

Herramienta para estimar la probabilidad de que un mercado cierre más arriba dentro de N barras, con datos públicos y sin bróker ni claves, y sobre todo para **medir si esa probabilidad sirve de algo**.

La idea central viene de `qbt`: un número sin su historial fuera de muestra no vale nada. Por eso cada modelo se compara con dos referencias tontas y el veredicto es severo a propósito. En mercados líquidos, lo normal es que diga «sin ventaja demostrada», y así lo dice.

**Demo en directo: [yayuuun.github.io/updown](https://yayuuun.github.io/updown/).** Dos de los modelos funcionan en tu navegador con BTC, ETH y SOL, cada probabilidad al lado del historial fuera de muestra de ese modelo.

*Herramienta de investigación, no asesoramiento financiero.*

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # tests sin red (datos sintéticos)
```

## Uso

```bash
python -m updown auditoria                                  # comprueba el evaluador con datos sintéticos
python -m updown evaluar SPY --intervalo 1d                 # walk-forward con los 13 modelos
python -m updown evaluar SPY,QQQ,IWM,DIA --intervalo 1h     # varios activos: un solo modelo con todos
python -m updown evaluar BTCUSDT --fuente binance --intervalo 1h
python -m updown evaluar SPY --modelos lineal,ridge,knn     # solo algunos modelos
python -m updown predecir <símbolo> --intervalo 1h          # P(sube) para la última barra cerrada
python -m updown vivo <símbolo> --fuente binance --intervalo 1m   # bucle + registro verificable
python -m updown evaluate SPY --interval 1d                  # los mismos comandos, con nombres en inglés
python -m estudios.comparativa                              # la comparativa de abajo
python -m estudios.etapas_confianza                         # confianza, etapas del HMM y modelos por etapa
python -m estudios.graficos                                 # gráficos del README a partir de los resultados
```

Opciones comunes: `--horizonte N` (barras hacia delante), `--contexto SPY,^VIX` o `--sin-contexto`, y en `evaluar`, `--coste-bps` y `--bloques`. `evaluar` muestra además el acierto según confianza del mejor modelo.

## Web (botón de escenarios)

```bash
streamlit run app.py
```

Muestra en qué escenario está cada mercado ahora (noticia/sobresalto, crisis, calma sin noticias, alcista, bajista, lateral) y, debajo, las probabilidades del HMM, como descripción del momento y no como señal. Tres pestañas:

- **Por vela**: qué vela de la próxima hora se predice mejor y, con el botón de escenario, qué modelo funciona mejor en cada uno.
- **Ranking por etapa**: con el mismo botón, sobre las etapas por reglas o las del HMM, qué modelo eligió la validación en cada bloque, cómo quedó cada candidato fuera de muestra en esa etapa, si los dos rankings coinciden y si el modelo por etapas supera al único.
- **Confianza**: la curva de acierto según cuánto se atreve a predecir, con su margen de error.

Solo evaluación histórica. Necesita Streamlit 1.50 o posterior.

## Demo en el navegador (GitHub Pages)

La página estática de `web/` se publica en [yayuuun.github.io/updown](https://yayuuun.github.io/updown/) sin servidor detrás. El navegador descarga las últimas 1.000 velas de 15 minutos de BTC, ETH y SOL de Binance, quita la que sigue abierta, calcula las mismas variables que `updown/features.py` (copiadas línea a línea en `web/js/features.js`) y ejecuta `ridge` y `gradient_boosting`, que `updown/export.py` convierte en coeficientes y árboles.

- `python -m estudios.web_modelos` descarga unos dos años de velas, evalúa los dos modelos con el mismo walk-forward purgado y escribe `web/data/models.json` y `web/data/metrics.json`. El workflow de Pages lo repite cada lunes.
- `python -m estudios.web_resultados` pasa las tablas de `estudios/resultados_*.md` a `web/data/results.json`: ningún número de la página está escrito a mano.
- `tests/test_web.py` ejecuta el JavaScript con Node y comprueba que las variables y las probabilidades coinciden con las de Python.

Para probarla en local: `python -m http.server --directory web` y abre http://localhost:8000.

## Vela a vela y escenarios

- `dataset(..., candles=4)` añade la dirección de cada una de las 4 próximas velas por separado (`up_v1..up_v4`); `evaluate_candles` compara todos los modelos en cada vela y en la hora completa. El transformer se entrena una vez con las 5 salidas.
- `updown/regimes.py` etiqueta cada vela con su escenario usando solo el pasado: las noticias no se leen, se detecta su huella (movimientos de >4 sigmas o volumen anómalo). El escenario se ve cuando ya ha empezado, nunca antes.
- `python -m estudios.velas_regimenes` escribe `estudios/resultados_velas_regimenes.md`.

## Acierto según confianza

El AUC es la media de todas las velas, pero el modelo no está igual de seguro siempre: a veces dice 50,5 % y a veces 58 %. `updown/confidence.py` ordena las predicciones fuera de muestra por lo lejos que están del 50 % y mide el acierto del 100 %, 50 %, 20 %, 10 % y 5 % más seguras, con cuántas velas son y su margen de error (IC 95 % agrupado por día: tres activos a la misma hora no son tres apuestas independientes).

- El corte de «seguro» se decide solo con las probabilidades, nunca mirando si acertó; si no, el resultado saldría inflado.
- Otra versión fija el corte de cada bloque con las predicciones del bloque anterior, como habría que hacerlo en vivo, y dice qué parte de las velas pasó realmente ese corte.

## Etapas con probabilidad (HMM)

`updown/hmm.py` aprende de los propios datos que el mercado pasa por unos pocos estados ocultos (4 por defecto), cada uno con su volatilidad y su tendencia, y con probabilidades de pasar de uno a otro. Da probabilidades, del tipo «75 % calma, 20 % alcista, 5 % crisis», no una etiqueta.

- Observa en cada vela lo mismo que las reglas, sin escala: la volatilidad del último día frente a su último año y el retorno de 30 días dividido por la volatilidad de ese plazo.
- En cada bloque de la evaluación se entrena solo con el periodo de entrenamiento, y cada vela recibe su probabilidad **filtrada** (con la información hasta ese momento), nunca la suavizada, que se recalcula con lo que vino después.
- Cada estado se nombra por la zona en la que caen su volatilidad y su tendencia medias (crisis, volátil, calma, alcista, bajista, lateral); dos estados en la misma zona comparten nombre y su probabilidad se suma.
- La etapa solo cuenta si su probabilidad llega al 90 %; si no, la vela es «incierta» y la predice el modelo general.
- El estudio compara estas etapas con las de reglas (tabla cruzada y porcentaje de coincidencia).

Está escrito con numpy: `hmmlearn` no tiene ruedas para los Python recientes y sus probabilidades son suavizadas.

## Submodelos por etapa

`updown/stages.py` entrena, para cada modelo y cada etapa (por reglas o del HMM), un submodelo solo con velas pasadas de esa etapa. El modelo general (cada modelo entrenado con todas las velas) compite también, y se queda las etapas con pocas velas.

- Dentro del entrenamiento se separa el último 20 % como validación (con purga). Con el 80 % anterior se entrenan los candidatos y gana cada etapa el que mejor predice su validación (Brier). El ganador se vuelve a entrenar con todo el entrenamiento. El periodo de prueba nunca participa en la elección.
- Fuera de muestra se compara con el mismo procedimiento sin etapas (la comparación justa: separa especializarse de elegir modelo), con cada modelo único y el mejor a posteriori, y con las dos referencias. Cuenta como un intento más en la corrección de Holm, y la diferencia de Brier se contrasta con un test pareado.
- Dos rankings por etapa: qué eligió la validación y cómo quedó cada submodelo fuera de muestra. Si coinciden, la elección es estable; si no, la etapa es demasiado ruidosa para fiarse.
- El transformer solo compite como modelo general: uno por etapa y por bloque costaría horas.

`python -m estudios.etapas_confianza` responde a las tres preguntas con BTC, ETH y SOL en velas de 15 minutos y escribe `estudios/resultados_etapas_confianza.md` (más de una hora en un portátil de 4 núcleos; `--objetivos vela_1` lo reduce a la mitad; `--transformer` añade el transformer como candidato general).

## Datos

| Fuente | Qué | Profundidad |
|---|---|---|
| Yahoo Finance (`yfinance`) | acciones, ETF, índices, divisas, cripto (`AAPL`, `^GSPC`, `EURUSD=X`, `BTC-USD`) | 1m: 7 días · 5m-30m: 60 días · 1h: 730 días · 1d: todo |
| Binance (API pública) | cripto en tiempo real (`BTCUSDT`) | 20.000 barras por defecto (`BINANCE_DEFAULT_BARS` en `updown/data.py`) |

Solo se usan **barras cerradas**: la última suele estar abierta y predecir sobre ella es mirar el futuro a medias. Las barras diarias de bolsa se dan por cerradas 20 minutos después del cierre de sesión que publica Yahoo.

Donde `api.binance.com` no responde (p. ej. desde servidores en EE. UU.), `UPDOWN_BINANCE_URL=https://data-api.binance.vision/api/v3/klines` usa su espejo de solo datos de mercado.

Las velas de Binance traen además el **flujo de órdenes**: cuántas operaciones hubo y qué parte del volumen fueron compras agresivas (órdenes que compran al precio de venta en ese momento). La caché guarda una versión: una descarga antigua, sin esas columnas, se vuelve a pedir entera en vez de mezclarse con las nuevas.

Además del propio activo se añade **contexto**: SPY (y el VIX en diario) para acciones, BTC para cripto. Se alinea con la última barra de referencia ya cerrada; si los husos horarios difieren, con la anterior (conservador).

## Modelos

Dos referencias: `siempre_sube` (la frecuencia de subidas del entrenamiento) y `momentum` (P(sube | la última barra subió/bajó)).

Trece modelos, con los hiperparámetros fijados de antemano (ajustarlos mirando resultados sería probar más configuraciones):

| Familia | Modelos |
|---|---|
| Lineales | `lineal` (logística), `ridge`, `lasso`, `elastic_net`, `lda`, `svm_lineal` |
| Probabilísticos / vecindad | `naive_bayes`, `knn` (6 variables fijas: con 30 dimensiones se pierde) |
| No lineales | `random_forest`, `extra_trees`, `gradient_boosting`, `red_neuronal` (MLP 32-16) |
| Combinación | `ensemble` (media de ridge, lasso, random forest y gradient boosting) |

Unas 26 variables, todas adimensionales (retornos a 1-60 barras, volatilidades, RSI, MACD, Bollinger, ATR, posición en el rango, gap, volumen relativo, calendario y contexto). Al no tener escala, varios activos pueden entrenar **un solo modelo**: es la forma honesta de tener «mucha información» sin inventarla.

Con velas de Binance se añaden 4 de flujo de órdenes: qué parte del volumen fueron compras agresivas (0,5 es equilibrio), la misma proporción en la última hora, si esa proporción es inusual respecto al último día y si el número de operaciones es inusual. El `transformer` (opcional, PyTorch) recibe además la proporción de compras agresivas vela a vela, como un canal más.

## Cómo se evalúa

- **Walk-forward** con ventana creciente y purga de `horizonte` barras: cada bloque se predice con un modelo entrenado solo con el pasado.
- **Brier frente a `siempre_sube`**, no accuracy: si sube el 55 % de los días, decir siempre «sube» ya acierta el 55 %.
- **AUC con p-valor corregido por Holm** sobre los 13 modelos. Con 13 intentos, alguno sale «significativo» al 5 % por azar casi la mitad de las veces.
- **Consistencia:** ganar a la referencia en al menos 2/3 de los bloques.
- **Simulación con costes** (largo/fuera, 5 pb por operación) frente a comprar y mantener.
- **Auditoría:** con un paseo aleatorio el veredicto debe ser «sin ventaja», una señal plantada debe detectarse y un canario de fuga debe disparar el AUC. Si alguna falla, el evaluador no es fiable.
- **Registro en vivo:** `vivo` apunta cada predicción antes de conocer el resultado y luego la completa. Es la única prueba que no se puede retocar sin querer.
- **Tests que lo vigilan:** que las etapas probabilísticas no miren al futuro, que la elección por etapa no use el periodo de prueba (se cambia el futuro y la elección debe quedar igual) y que la curva de confianza no use los aciertos para el corte.

## Resultados

En `estudios/resultados_comparativa.md`: SPY diario desde 1993, 20 grandes empresas agrupadas, 10 ETF por horas y BTCUSDT por horas. Vela a vela y por régimen, en `estudios/resultados_velas_regimenes.md`. Esos resultados son anteriores al flujo de órdenes y a fijar la semilla de lasso; al repetir los estudios los números se moverán algo.

`estudios/resultados_etapas_confianza.md` aparecerá al ejecutar `python -m estudios.etapas_confianza`: todavía no hay resultados publicados de la confianza ni de los modelos por etapa.

## Limitaciones

- El universo de «grandes empresas» son las grandes de hoy: **sesgo de supervivencia**, así que infla cualquier resultado positivo.
- Los datos intradía de Yahoo pueden llegar con retraso según la bolsa, y el máximo histórico es corto (730 días en 1h).
- El cierre de sesión diario se ha probado con datos sintéticos; con datos reales, compruébalo antes de fiarte del modo `vivo` en diario.
- Nada medido aquí garantiza que un patrón siga funcionando en el próximo régimen de mercado.
- La probabilidad del HMM es la del propio modelo, no una frecuencia comprobada: un estado oculto no tiene verdad con la que contrastarlo. Con observaciones suavizadas suele estar cerca del 100 %, así que la regla del 90 % puede filtrar poco; el estudio dice qué parte de las velas quedó «incierta».
- La elección por etapa es ruidosa cuando una etapa tiene pocas velas de validación; la coincidencia entre el ranking de validación y el de fuera de muestra dice cuánto fiarse.
