# updown: ¿subirá?, con su historial honesto al lado

Herramienta para estimar la probabilidad de que un mercado cierre más arriba dentro de N barras, con datos públicos y sin bróker ni claves, y sobre todo para **medir si esa probabilidad sirve de algo**.

La idea central viene de `qbt`: un número sin su historial fuera de muestra no vale nada. Por eso cada modelo se compara con dos referencias tontas y el veredicto es severo a propósito. En mercados líquidos, lo normal es que diga «sin ventaja demostrada», y así lo dice.

*Herramienta de investigación, no asesoramiento financiero.*

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # 32 tests
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
python -m estudios.comparativa                              # la comparativa de abajo
```

Opciones comunes: `--horizonte N` (barras hacia delante), `--contexto SPY,^VIX` o `--sin-contexto`, y en `evaluar`, `--coste-bps` y `--bloques`.

## Web (botón de escenarios)

```bash
streamlit run app.py
```

Muestra en qué escenario está cada mercado ahora (noticia/sobresalto, crisis, calma sin noticias, alcista, bajista, lateral), qué vela de la próxima hora se predice mejor y, con el botón de escenario, qué modelo funciona mejor en cada uno. Solo evaluación histórica.

## Vela a vela y escenarios

- `dataset(..., candles=4)` añade la dirección de cada una de las 4 próximas velas por separado (`up_v1..up_v4`); `evaluate_candles` compara todos los modelos en cada vela y en la hora completa. El transformer se entrena una vez con las 5 salidas.
- `updown/regimes.py` etiqueta cada vela con su escenario usando solo el pasado: las noticias no se leen, se detecta su huella (movimientos de >4 sigmas o volumen anómalo). El escenario se ve cuando ya ha empezado, nunca antes.
- `python -m estudios.velas_regimenes` escribe `estudios/resultados_velas_regimenes.md`.

## Datos

| Fuente | Qué | Profundidad |
|---|---|---|
| Yahoo Finance (`yfinance`) | acciones, ETF, índices, divisas, cripto (`AAPL`, `^GSPC`, `EURUSD=X`, `BTC-USD`) | 1m: 7 días · 5m-30m: 60 días · 1h: 730 días · 1d: todo |
| Binance (API pública) | cripto en tiempo real (`BTCUSDT`) | 20.000 barras por defecto (`BINANCE_DEFAULT_BARS` en `updown/data.py`) |

Solo se usan **barras cerradas**: la última suele estar abierta y predecir sobre ella es mirar el futuro a medias. Las barras diarias de bolsa se dan por cerradas 20 minutos después del cierre de sesión que publica Yahoo.

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

## Cómo se evalúa

- **Walk-forward** con ventana creciente y purga de `horizonte` barras: cada bloque se predice con un modelo entrenado solo con el pasado.
- **Brier frente a `siempre_sube`**, no accuracy: si sube el 55 % de los días, decir siempre «sube» ya acierta el 55 %.
- **AUC con p-valor corregido por Holm** sobre los 13 modelos. Con 13 intentos, alguno sale «significativo» al 5 % por azar casi la mitad de las veces.
- **Consistencia:** ganar a la referencia en al menos 2/3 de los bloques.
- **Simulación con costes** (largo/fuera, 5 pb por operación) frente a comprar y mantener.
- **Auditoría:** con un paseo aleatorio el veredicto debe ser «sin ventaja», una señal plantada debe detectarse y un canario de fuga debe disparar el AUC. Si alguna falla, el evaluador no es fiable.
- **Registro en vivo:** `vivo` apunta cada predicción antes de conocer el resultado y luego la completa. Es la única prueba que no se puede retocar sin querer.

## Resultados

En `estudios/resultados_comparativa.md`: SPY diario desde 1993, 20 grandes empresas agrupadas, 10 ETF por horas y BTCUSDT por horas.

## Limitaciones

- El universo de «grandes empresas» son las grandes de hoy: **sesgo de supervivencia**, así que infla cualquier resultado positivo.
- Los datos intradía de Yahoo pueden llegar con retraso según la bolsa, y el máximo histórico es corto (730 días en 1h).
- El cierre de sesión diario se ha probado con datos sintéticos; con datos reales, compruébalo antes de fiarte del modo `vivo` en diario.
- Nada medido aquí garantiza que un patrón siga funcionando en el próximo régimen de mercado.
