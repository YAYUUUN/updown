# Vela a vela y por régimen (15 min, BTC/ETH/SOL)

Generado 2026-10-22 12:59 UTC · 209751 filas · fuera de muestra 2025-08-11 → 2026-10-22 · sin costes · Holm por objetivo.

## AUC por vela

| modelo | vela_1 | vela_2 | vela_3 | vela_4 | hora |
|---|---|---|---|---|---|
| gradient_boosting | 0.5418 | 0.523 | 0.5192 | 0.5141 | 0.5344 |
| momentum | 0.522 | 0.5109 | 0.5056 | 0.5006 | 0.517 |
| random_forest | 0.5424 | 0.5245 | 0.5195 | 0.5168 | 0.5372 |
| ridge | 0.5401 | 0.5205 | 0.5158 | 0.5131 | 0.5264 |
| siempre_sube | 0.5021 | 0.502 | 0.502 | 0.502 | 0.5008 |
| transformer | 0.5402 | 0.5223 | 0.5158 | 0.5131 | 0.5301 |

## Acierto por vela

| modelo | vela_1 | vela_2 | vela_3 | vela_4 | hora |
|---|---|---|---|---|---|
| gradient_boosting | 0.5289 | 0.5169 | 0.5142 | 0.5098 | 0.5215 |
| momentum | 0.5237 | 0.5124 | 0.5052 | 0.501 | 0.5166 |
| random_forest | 0.5294 | 0.517 | 0.514 | 0.5115 | 0.5239 |
| ridge | 0.5291 | 0.515 | 0.5131 | 0.5117 | 0.5173 |
| siempre_sube | 0.4967 | 0.4967 | 0.4967 | 0.4967 | 0.5015 |
| transformer | 0.5296 | 0.516 | 0.5104 | 0.5095 | 0.52 |

## ¿Ventaja demostrada? (Holm < 0.05, mejora el Brier y gana en ≥ 2/3 de bloques)

| modelo | vela_1 | vela_2 | vela_3 | vela_4 | hora |
|---|---|---|---|---|---|
| gradient_boosting | 1 | 1 | 1 | 1 | 1 |
| momentum | 0 | 0 | 0 | 0 | 0 |
| random_forest | 1 | 1 | 1 | 1 | 1 |
| ridge | 1 | 1 | 1 | 1 | 1 |
| siempre_sube | 0 | 0 | 0 | 0 | 0 |
| transformer | 1 | 1 | 1 | 1 | 1 |

## Tiempo en cada régimen

| modelo | tiempo |
|---|---|
| alcista | 0.2718 |
| bajista | 0.2194 |
| calma | 0.2049 |
| crisis | 0.07189 |
| lateral | 0.07409 |
| shock | 0.158 |

## Por régimen: vela_1

| modelo | n | freq_sube | acierto | auc | skill_vs_siempre_sube |
|---|---|---|---|---|---|
| shock · gradient_boosting | 2.01e+04 | 0.4989 | 0.5215 | 0.5315 | 0.003142 |
| shock · transformer | 2.01e+04 | 0.4989 | 0.5213 | 0.5299 | 0.00161 |
| shock · random_forest | 2.01e+04 | 0.4989 | 0.5209 | 0.5295 | 0.002788 |
| shock · ridge | 2.01e+04 | 0.4989 | 0.5125 | 0.5213 | -0.001034 |
| shock · momentum | 2.01e+04 | 0.4989 | 0.5183 | 0.5156 | 0.001198 |
| crisis · transformer | 9760 | 0.4888 | 0.5307 | 0.5396 | 0.004527 |
| crisis · ridge | 9760 | 0.4888 | 0.5269 | 0.5369 | 0.004217 |
| crisis · gradient_boosting | 9760 | 0.4888 | 0.5209 | 0.5343 | 0.002764 |
| crisis · random_forest | 9760 | 0.4888 | 0.5225 | 0.5343 | 0.002985 |
| crisis · momentum | 9760 | 0.4888 | 0.5206 | 0.525 | 0.00178 |
| calma · ridge | 2.811e+04 | 0.4971 | 0.5333 | 0.5469 | 0.006716 |
| calma · random_forest | 2.811e+04 | 0.4971 | 0.5314 | 0.5464 | 0.006607 |
| calma · gradient_boosting | 2.811e+04 | 0.4971 | 0.5309 | 0.5452 | 0.00673 |
| calma · transformer | 2.811e+04 | 0.4971 | 0.5301 | 0.5411 | 0.005066 |
| calma · momentum | 2.811e+04 | 0.4971 | 0.5277 | 0.5245 | 0.002635 |
| alcista · random_forest | 3.095e+04 | 0.4977 | 0.5303 | 0.5435 | 0.005948 |
| alcista · transformer | 3.095e+04 | 0.4977 | 0.5331 | 0.5421 | 0.00536 |
| alcista · gradient_boosting | 3.095e+04 | 0.4977 | 0.5288 | 0.5414 | 0.005834 |
| alcista · ridge | 3.095e+04 | 0.4977 | 0.5302 | 0.5405 | 0.005038 |
| alcista · momentum | 3.095e+04 | 0.4977 | 0.5215 | 0.5184 | 0.001698 |
| bajista · random_forest | 2.716e+04 | 0.4998 | 0.5362 | 0.5476 | 0.006218 |
| bajista · ridge | 2.716e+04 | 0.4998 | 0.5339 | 0.5462 | 0.005968 |
| bajista · gradient_boosting | 2.716e+04 | 0.4998 | 0.5342 | 0.5462 | 0.006092 |
| bajista · transformer | 2.715e+04 | 0.4997 | 0.532 | 0.5449 | 0.005904 |
| bajista · momentum | 2.716e+04 | 0.4998 | 0.5276 | 0.5286 | 0.002726 |
| lateral · gradient_boosting | 9765 | 0.4881 | 0.5321 | 0.5531 | 0.008822 |
| lateral · random_forest | 9765 | 0.4881 | 0.5261 | 0.5511 | 0.007838 |
| lateral · ridge | 9765 | 0.4881 | 0.5363 | 0.5486 | 0.007308 |
| lateral · transformer | 9765 | 0.4881 | 0.5265 | 0.5445 | 0.00659 |
| lateral · momentum | 9765 | 0.4881 | 0.5226 | 0.5212 | 0.001946 |

## Por régimen: hora

| modelo | n | freq_sube | acierto | auc | skill_vs_siempre_sube |
|---|---|---|---|---|---|
| shock · random_forest | 2.01e+04 | 0.5046 | 0.5191 | 0.5325 | 0.00298 |
| shock · gradient_boosting | 2.01e+04 | 0.5046 | 0.5179 | 0.5308 | 0.002517 |
| shock · transformer | 2.01e+04 | 0.5046 | 0.5165 | 0.5245 | -0.001037 |
| shock · ridge | 2.01e+04 | 0.5046 | 0.5104 | 0.5154 | 0.0004643 |
| shock · momentum | 2.01e+04 | 0.5046 | 0.5129 | 0.5091 | 0.0006221 |
| crisis · transformer | 9760 | 0.4768 | 0.5062 | 0.5318 | 0.0022 |
| crisis · momentum | 9760 | 0.4768 | 0.5168 | 0.5259 | 0.001079 |
| crisis · gradient_boosting | 9760 | 0.4768 | 0.5073 | 0.5247 | -0.0005262 |
| crisis · random_forest | 9760 | 0.4768 | 0.4998 | 0.5228 | -0.0004697 |
| crisis · ridge | 9760 | 0.4768 | 0.4966 | 0.5158 | 0.001264 |
| calma · random_forest | 2.811e+04 | 0.5004 | 0.5254 | 0.538 | 0.004274 |
| calma · gradient_boosting | 2.811e+04 | 0.5004 | 0.5232 | 0.537 | 0.004567 |
| calma · transformer | 2.811e+04 | 0.5004 | 0.5207 | 0.5305 | 0.002518 |
| calma · ridge | 2.811e+04 | 0.5004 | 0.521 | 0.5292 | 0.002826 |
| calma · momentum | 2.811e+04 | 0.5004 | 0.5153 | 0.5178 | 0.0009381 |
| alcista · random_forest | 3.096e+04 | 0.5057 | 0.5275 | 0.5383 | 0.004512 |
| alcista · transformer | 3.096e+04 | 0.5057 | 0.5245 | 0.5348 | 0.003763 |
| alcista · gradient_boosting | 3.096e+04 | 0.5057 | 0.5229 | 0.5321 | 0.003413 |
| alcista · ridge | 3.096e+04 | 0.5057 | 0.5196 | 0.5281 | 0.002396 |
| alcista · momentum | 3.096e+04 | 0.5057 | 0.5207 | 0.5188 | 0.001605 |
| bajista · random_forest | 2.715e+04 | 0.5103 | 0.5312 | 0.5438 | 0.00482 |
| bajista · gradient_boosting | 2.715e+04 | 0.5103 | 0.5273 | 0.5416 | 0.005445 |
| bajista · ridge | 2.715e+04 | 0.5103 | 0.5236 | 0.5336 | 0.002349 |
| bajista · transformer | 2.715e+04 | 0.5103 | 0.5228 | 0.5324 | 0.00358 |
| bajista · momentum | 2.715e+04 | 0.5103 | 0.5148 | 0.519 | 0.0008449 |
| lateral · random_forest | 9765 | 0.4845 | 0.5211 | 0.5497 | 0.006662 |
| lateral · gradient_boosting | 9765 | 0.4845 | 0.5172 | 0.5408 | 0.00635 |
| lateral · ridge | 9765 | 0.4845 | 0.5172 | 0.5387 | 0.004446 |
| lateral · transformer | 9765 | 0.4845 | 0.5171 | 0.5322 | 0.003705 |
| lateral · momentum | 9765 | 0.4845 | 0.5204 | 0.5185 | 0.001526 |

*Herramienta de investigación, no asesoramiento financiero.*
