# Comparativa de modelos

Generado 2026-10-22 00:06 UTC. Walk-forward de 6 bloques, coste 5 pb por operación, p-valores corregidos por Holm sobre 13 modelos.

## SPY diario (desde 1993)

8429 filas, 26 variables; fuera de muestra de 2006-09-12 a 2026-10-19.

| modelo | acierto | freq_sube | auc | p_auc_corregido | skill_vs_siempre_sube | bloques_ganados | ret_anual_estrategia | ret_anual_comprar_y_mantener |
|---|---|---|---|---|---|---|---|---|
| svm_lineal | 0.5511 | 0.5509 | 0.4913 | 1 | 0.0001781 | 3 | 0.1103 | 0.112 |
| ridge | 0.5509 | 0.5509 | 0.4918 | 1 | 3.564e-05 | 3 | 0.1022 | 0.112 |
| siempre_sube | 0.5509 | 0.5509 | 0.4969 |  | 0 | 0 | 0.112 | 0.112 |
| momentum | 0.5509 | 0.5509 | 0.5056 |  | -1.39e-06 | 3 | 0.1065 | 0.112 |
| extra_trees | 0.5521 | 0.5509 | 0.5052 | 1 | -0.0004364 | 3 | 0.08532 | 0.112 |
| ensemble | 0.5393 | 0.5509 | 0.5033 | 1 | -0.002054 | 1 | 0.06518 | 0.112 |
| lasso | 0.5396 | 0.5509 | 0.505 | 1 | -0.003071 | 1 | 0.07531 | 0.112 |
| random_forest | 0.5379 | 0.5509 | 0.5039 | 1 | -0.003509 | 1 | 0.07184 | 0.112 |
| elastic_net | 0.5383 | 0.5509 | 0.5037 | 1 | -0.004833 | 0 | 0.07019 | 0.112 |
| lda | 0.5288 | 0.5509 | 0.4983 | 1 | -0.01051 | 0 | 0.02541 | 0.112 |
| lineal | 0.5268 | 0.5509 | 0.4984 | 1 | -0.01207 | 0 | 0.02459 | 0.112 |
| gradient_boosting | 0.5254 | 0.5509 | 0.5082 | 1 | -0.01438 | 1 | 0.01509 | 0.112 |
| knn | 0.5165 | 0.5509 | 0.4963 | 1 | -0.0165 | 0 | 0.01525 | 0.112 |
| red_neuronal | 0.5211 | 0.5509 | 0.5037 | 1 | -0.02671 | 0 | 0.04276 | 0.112 |
| naive_bayes | 0.495 | 0.5509 | 0.5059 | 1 | -0.1866 | 0 | 0.051 | 0.112 |

**Veredicto:** SIN VENTAJA DEMOSTRADA. Ningún modelo predice mejor que 'siempre_sube' de forma significativa y consistente. Sus probabilidades no son una señal: no las uses para operar.

## 20 grandes empresas de EE. UU., diario, agrupadas

165304 filas, 29 variables; fuera de muestra de 2006-08-08 a 2026-10-19.

| modelo | acierto | freq_sube | auc | p_auc_corregido | skill_vs_siempre_sube | bloques_ganados | ret_anual_estrategia | ret_anual_comprar_y_mantener |
|---|---|---|---|---|---|---|---|---|
| svm_lineal | 0.5194 | 0.5175 | 0.5119 | 1.68e-10 | 0.001505 | 6 | 0.05555 | 0.1254 |
| ridge | 0.5194 | 0.5175 | 0.5119 | 1.68e-10 | 0.001503 | 6 | 0.05582 | 0.1254 |
| ensemble | 0.5101 | 0.5175 | 0.5133 | 8.756e-13 | 0.0008952 | 6 | 0.04874 | 0.1254 |
| extra_trees | 0.5099 | 0.5175 | 0.5138 | 1.063e-13 | 0.0007875 | 5 | 0.03618 | 0.1254 |
| random_forest | 0.5059 | 0.5175 | 0.5118 | 1.68e-10 | 0.0003969 | 4 | 0.0394 | 0.1254 |
| momentum | 0.5075 | 0.5175 | 0.5113 |  | 0.0001983 | 4 | 0 | 0.1254 |
| siempre_sube | 0.5061 | 0.5175 | 0.5061 |  | 0 | 0 | 0 | 0.1254 |
| lasso | 0.5088 | 0.5175 | 0.5144 | 1.198e-14 | -7.89e-05 | 3 | 0.04171 | 0.1254 |
| elastic_net | 0.5093 | 0.5175 | 0.5145 | 5.564e-15 | -0.0001156 | 3 | 0.03921 | 0.1254 |
| lda | 0.5103 | 0.5175 | 0.5147 | 3.32e-15 | -0.0001628 | 3 | 0.03789 | 0.1254 |
| lineal | 0.51 | 0.5175 | 0.5146 | 5.02e-15 | -0.0002044 | 3 | 0.03868 | 0.1254 |
| gradient_boosting | 0.5079 | 0.5175 | 0.5094 | 2.188e-07 | -0.002065 | 1 | 0.02269 | 0.1254 |
| knn | 0.5034 | 0.5175 | 0.5055 | 0.001247 | -0.003271 | 0 | 0.02137 | 0.1254 |
| red_neuronal | 0.5074 | 0.5175 | 0.5122 | 5.171e-11 | -0.02251 | 0 | 0.02152 | 0.1254 |
| naive_bayes | 0.5051 | 0.5175 | 0.511 | 1.728e-09 | -0.07841 | 0 | 0.04212 | 0.1254 |

**Veredicto:** POSIBLE VENTAJA ESTADÍSTICA en 'svm_lineal' (AUC 0.512, p corregido 1.68e-10, gana en 6/6 bloques). Pero tras costes NO supera a comprar y mantener, y nada de esto garantiza que siga funcionando. Valídalo en vivo antes de creértelo.

## 10 ETF, 1 hora, agrupados (730 días)

50220 filas, 24 variables; fuera de muestra de 2025-01-31 a 2026-10-21.

| modelo | acierto | freq_sube | auc | p_auc_corregido | skill_vs_siempre_sube | bloques_ganados | ret_anual_estrategia | ret_anual_comprar_y_mantener |
|---|---|---|---|---|---|---|---|---|
| siempre_sube | 0.5158 | 0.5158 | 0.5002 |  | 0 | 0 | 0.1534 | 0.1538 |
| momentum | 0.5158 | 0.5158 | 0.5026 |  | -7.159e-06 | 3 | -0.2447 | 0.1538 |
| extra_trees | 0.5158 | 0.5158 | 0.5071 | 0.2009 | -0.0001732 | 2 | -0.08869 | 0.1538 |
| random_forest | 0.5144 | 0.5158 | 0.5101 | 0.01576 | -0.0007228 | 3 | -0.1313 | 0.1538 |
| svm_lineal | 0.5149 | 0.5158 | 0.4944 | 1 | -0.001087 | 1 | -0.1299 | 0.1538 |
| ridge | 0.5147 | 0.5158 | 0.4945 | 1 | -0.001112 | 1 | -0.1311 | 0.1538 |
| ensemble | 0.5131 | 0.5158 | 0.5039 | 1 | -0.001209 | 3 | -0.1389 | 0.1538 |
| lasso | 0.5124 | 0.5158 | 0.5029 | 1 | -0.002238 | 3 | -0.1231 | 0.1538 |
| elastic_net | 0.512 | 0.5158 | 0.502 | 1 | -0.002912 | 3 | -0.1286 | 0.1538 |
| lda | 0.5114 | 0.5158 | 0.5012 | 1 | -0.004211 | 3 | -0.1408 | 0.1538 |
| lineal | 0.5112 | 0.5158 | 0.5011 | 1 | -0.004397 | 3 | -0.1405 | 0.1538 |
| gradient_boosting | 0.5142 | 0.5158 | 0.5034 | 1 | -0.00477 | 0 | -0.1953 | 0.1538 |
| knn | 0.5084 | 0.5158 | 0.5022 | 1 | -0.006488 | 0 | -0.2779 | 0.1538 |
| red_neuronal | 0.51 | 0.5158 | 0.5068 | 0.226 | -0.02091 | 0 | -0.2095 | 0.1538 |
| naive_bayes | 0.5074 | 0.5158 | 0.5025 | 1 | -0.07136 | 0 | -0.06157 | 0.1538 |

**Veredicto:** SIN VENTAJA DEMOSTRADA. Ningún modelo predice mejor que 'siempre_sube' de forma significativa y consistente. Sus probabilidades no son una señal: no las uses para operar.

## BTCUSDT, 1 hora, Binance (20.000 barras)

19939 filas, 24 variables; fuera de muestra de 2025-06-10 a 2026-10-21.

| modelo | acierto | freq_sube | auc | p_auc_corregido | skill_vs_siempre_sube | bloques_ganados | ret_anual_estrategia | ret_anual_comprar_y_mantener |
|---|---|---|---|---|---|---|---|---|
| ensemble | 0.5242 | 0.5005 | 0.5341 | 5.697e-10 | 0.003742 | 6 | -0.6016 | -0.177 |
| random_forest | 0.5247 | 0.5005 | 0.5352 | 1.666e-10 | 0.003706 | 6 | -0.6246 | -0.177 |
| extra_trees | 0.523 | 0.5005 | 0.5345 | 3.882e-10 | 0.003316 | 6 | -0.5586 | -0.177 |
| ridge | 0.5195 | 0.5005 | 0.5274 | 6.136e-07 | 0.002252 | 5 | -0.4187 | -0.177 |
| svm_lineal | 0.5196 | 0.5005 | 0.5271 | 6.136e-07 | 0.002213 | 6 | -0.4166 | -0.177 |
| lasso | 0.5204 | 0.5005 | 0.5278 | 5.531e-07 | 0.002051 | 6 | -0.6468 | -0.177 |
| elastic_net | 0.5188 | 0.5005 | 0.527 | 6.136e-07 | 0.001852 | 5 | -0.6697 | -0.177 |
| lda | 0.5214 | 0.5005 | 0.5274 | 6.136e-07 | 0.00166 | 4 | -0.606 | -0.177 |
| lineal | 0.5222 | 0.5005 | 0.5276 | 5.723e-07 | 0.001604 | 4 | -0.6125 | -0.177 |
| momentum | 0.5167 | 0.5005 | 0.5123 |  | 0.0008842 | 4 | -0.91 | -0.177 |
| gradient_boosting | 0.5243 | 0.5005 | 0.5312 | 1.738e-08 | 0.0008318 | 4 | -0.6448 | -0.177 |
| siempre_sube | 0.5005 | 0.5005 | 0.4965 |  | 0 | 0 | 0 | -0.177 |
| red_neuronal | 0.5154 | 0.5005 | 0.524 | 5.546e-06 | -0.003182 | 2 | -0.6595 | -0.177 |
| knn | 0.5157 | 0.5005 | 0.5225 | 9.932e-06 | -0.003854 | 1 | -0.7947 | -0.177 |
| naive_bayes | 0.5219 | 0.5005 | 0.5303 | 4.407e-08 | -0.1099 | 0 | -0.5534 | -0.177 |

**Veredicto:** POSIBLE VENTAJA ESTADÍSTICA en 'ensemble' (AUC 0.534, p corregido 5.7e-10, gana en 6/6 bloques). Pero tras costes NO supera a comprar y mantener, y nada de esto garantiza que siga funcionando. Valídalo en vivo antes de creértelo.

*Herramienta de investigación, no asesoramiento financiero.*
