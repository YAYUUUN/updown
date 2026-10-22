# Transformer vs. clásicos: P(sube en 1 hora), barras de 15 min

Generado 2026-10-22 12:21 UTC. BTCUSDT, ETHUSDT, SOLUSDT agrupados, 209751 filas; fuera de muestra de 2025-08-11 a 2026-10-22. Horizonte 4 barras, ventana 64 barras, 6 bloques walk-forward, Holm sobre 5 modelos, p-valores con n efectivo = n/4 por el solape de objetivos.

| modelo | acierto | freq_sube | auc | p_auc_corregido | skill_vs_siempre_sube | bloques_ganados | bloques |
|---|---|---|---|---|---|---|---|
| random_forest | 0.5242 | 0.5015 | 0.5373 | 6.278e-30 | 0.004115 | 6 | 6 |
| gradient_boosting | 0.5213 | 0.5015 | 0.5343 | 1.084e-25 | 0.003861 | 6 | 6 |
| lasso | 0.5198 | 0.5015 | 0.5285 | 2.756e-18 | 0.002413 | 6 | 6 |
| ridge | 0.5174 | 0.5015 | 0.5264 | 4.64e-16 | 0.002249 | 6 | 6 |
| transformer | 0.516 | 0.5015 | 0.5243 | 3.808e-14 | 0.001656 | 5 | 6 |
| momentum | 0.5166 | 0.5015 | 0.517 |  | 0.001087 | 6 | 6 |
| siempre_sube | 0.5015 | 0.5015 | 0.5008 |  | 0 | 0 | 6 |

Brier por bloque:

| modelo | siempre_sube | momentum | ridge | lasso | random_forest | gradient_boosting | transformer |
|---|---|---|---|---|---|---|---|
| 0 | 0.25 | 0.2498 | 0.2498 | 0.2497 | 0.2495 | 0.2495 | 0.25 |
| 1 | 0.2499 | 0.2497 | 0.2498 | 0.2499 | 0.2493 | 0.2494 | 0.2505 |
| 2 | 0.2502 | 0.2499 | 0.2498 | 0.2497 | 0.2491 | 0.249 | 0.2495 |
| 3 | 0.2501 | 0.2498 | 0.2491 | 0.249 | 0.2487 | 0.2491 | 0.2492 |
| 4 | 0.2501 | 0.25 | 0.2493 | 0.2493 | 0.2489 | 0.249 | 0.2496 |
| 5 | 0.2499 | 0.2494 | 0.2492 | 0.249 | 0.2485 | 0.2485 | 0.249 |

**Veredicto:** POSIBLE VENTAJA ESTADÍSTICA en 'random_forest' (AUC 0.537, p corregido 6.28e-30, gana en 6/6 bloques). La simulación sin costes supera a comprar y mantener, y nada de esto garantiza que siga funcionando. Valídalo en vivo antes de creértelo.

*Herramienta de investigación, no asesoramiento financiero.*
