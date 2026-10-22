"""Submodelos por etapa con elección del mejor, sin tocar el periodo de prueba.

Pregunta: ¿especializarse por etapa mejora la predicción, o solo reparte los datos
en trozos más pequeños?

En cada bloque del walk-forward:

  1. Etapa de cada vela, con datos pasados. Dos fuentes:
       · reglas : el régimen de `regimes.py` (columna `regimen`).
       · hmm    : la etapa del HMM ajustado SOLO con el entrenamiento, si su
                  probabilidad filtrada llega al umbral; si no, «incierta».
     (Cualquier otra fuente es el nombre de una columna con etiquetas, que deben
     calcularse solo con el pasado.)
  2. Dentro del entrenamiento se separa el último 20 % como validación (con la
     misma purga que entre entrenamiento y prueba). Con el 80 % anterior se
     entrenan los candidatos de cada etapa:
       · cada modelo entrenado solo con las velas de esa etapa (`ridge@calma`);
       · cada modelo entrenado con todas las velas (`ridge@general`): el modelo
         general también compite.
     Gana la etapa el candidato con menor Brier en las velas de validación de esa
     etapa. Si la etapa tiene pocas velas (en entrenamiento o en validación), se
     queda el general: el modelo entrenado con todo que mejor predice la
     validación completa (`general_validado`).
  3. Los ganadores se vuelven a entrenar con TODO el entrenamiento. Cada vela de
     prueba la predice el ganador de su etapa; las inciertas o sin etapa, el general.

El periodo de prueba nunca participa en la elección. El HMM que etiqueta la
validación se ajusta solo con el 80 % anterior, para que esas etiquetas tampoco
hayan visto las velas que se usan para elegir.

Comparaciones fuera de muestra:

  · general_validado: el mismo procedimiento sin etapas. Es la comparación justa,
    porque separa el efecto de especializarse del de elegir modelo.
  · cada modelo único, y el mejor de ellos A POSTERIORI (elegido mirando la
    prueba, lo que le favorece: si el modelo por etapas le gana, es de verdad).
  · las referencias siempre_sube y momentum.

El modelo por etapas cuenta como un intento más en la corrección de Holm (y
general_validado, otro). La diferencia de Brier con cada rival se contrasta con un
test pareado (Diebold-Mariano) con error estándar agrupado por día.

Rankings por etapa: qué eligió la validación en cada bloque y cómo quedó cada
candidato fuera de muestra en las velas de esa etapa. Si coinciden, la elección
es estable; si no, la etapa es demasiado ruidosa para fiarse de ella.

El transformer, por defecto, solo compite como candidato general: entrenar uno
por etapa y por bloque costaría horas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluate import holm, scores, se_by_day, time_folds
from .features import model_columns
from .hmm import HMM_COLUMNS, UNCERTAIN, StageHMM, agreement, compare_with_rules
from .models import BASELINES, DEEP, make_model

STAGE_MODELS = ("ridge", "lasso", "knn", "random_forest", "gradient_boosting")
SOURCES = ("reglas", "hmm")
GENERAL = "general"


@dataclass
class StageReport:
    target: str
    metrics: pd.DataFrame          # una fila por método, fuera de muestra
    comparison: pd.DataFrame       # por etapas frente a general_validado y al mejor único (test pareado)
    choices: pd.DataFrame          # ganador de cada etapa en cada bloque y por qué
    val_rank: pd.DataFrame         # Brier de cada candidato en la validación de cada etapa y bloque
    oos_rank: pd.DataFrame         # cada candidato fuera de muestra en las velas de su etapa
    agreement: pd.DataFrame        # ¿la validación ordena igual que la prueba?
    stage_share: pd.DataFrame      # parte de las velas de prueba en cada etapa
    hmm_states: pd.DataFrame       # características de los estados del HMM en cada bloque
    hmm_vs_rules: pd.DataFrame     # de cada etapa del HMM, qué parte cae en cada régimen de reglas
    oos: dict = field(repr=False, default_factory=dict)
    meta: dict = field(default_factory=dict)


def inner_split(times: pd.Index, val_frac: float = 0.2, purge: int = 1):
    """Instantes de ajuste y de validación dentro del entrenamiento: el último `val_frac`, con purga."""
    uniq = pd.Index(times.unique()).sort_values()
    cut = int(len(uniq) * (1 - val_frac))
    return uniq[:max(cut - purge, 0)], uniq[cut:]


def _brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def _trainable(y: np.ndarray, min_train: int) -> bool:
    return len(y) >= min_train and len(np.unique(y)) == 2


def _stage_names(*labels: np.ndarray) -> list[str]:
    seen = set()
    for lab in labels:
        seen |= {s for s in pd.unique(lab) if isinstance(s, str) and s != UNCERTAIN}
    return sorted(seen)


def evaluate_stages(data: pd.DataFrame, models=STAGE_MODELS, target: str = "up", ret_col: str = "fwd_ret",
                    purge: int | None = None, sources=SOURCES, general_only=DEEP, n_folds: int = 6,
                    min_train_frac: float = 0.4, val_frac: float = 0.2, min_train: int = 1000,
                    min_val: int = 500, n_states: int = 4, threshold: float = 0.9,
                    verbose: bool = False) -> StageReport:
    """Walk-forward del modelo por etapas (una por fuente) frente a los modelos únicos.

    `data` debe crearse con regimes=True. Los modelos de `general_only` que estén en
    `models` solo se entrenan con todas las velas (nunca por etapa).
    """
    purge = purge or data.attrs.get("horizon", 1)
    if "reglas" in sources and "regimen" not in data:
        raise ValueError("Crea el dataset con regimes=True para usar las etapas por reglas.")
    if "hmm" in sources and not set(HMM_COLUMNS) <= set(data.columns):
        raise ValueError("Crea el dataset con regimes=True para usar las etapas del HMM.")
    fixed = {src: "regimen" if src == "reglas" else src for src in sources if src != "hmm"}
    missing = [c for c in fixed.values() if c not in data]
    if missing:
        raise ValueError(f"Faltan las columnas de etapas {missing}.")
    general_models = [m for m in models if m not in BASELINES]
    stage_models = [m for m in general_models if m not in general_only]
    cols = {m: model_columns(data, make_model(m)) for m in list(BASELINES) + general_models}
    work = data.assign(_pos=np.arange(len(data)))
    labeled = work[work[target].notna()]
    rules = work["regimen"].to_numpy(object) if "regimen" in work else None
    fixed_labels = {src: work[col].to_numpy(object) for src, col in fixed.items()}
    keep = [c for c in ("symbol", "regimen") if c in labeled]

    def fit_predict(name, train, test):
        if len(test) == 0:
            return np.empty(0)
        model = make_model(name).fit(train[cols[name]], train[target])
        return np.clip(np.asarray(model.proba(test[cols[name]]), dtype=float), 1e-4, 1 - 1e-4)

    oos_parts: dict[str, list] = {}
    choice_rows, val_rows, cand_parts, hmm_rows = [], [], [], []
    hmm_oos = np.full(len(work), np.nan, dtype=object)
    folds = list(time_folds(labeled.index, n_folds, min_train_frac, purge))
    if not folds:
        raise ValueError("No hay histórico suficiente para evaluar.")
    for i, (train_t, test_t) in enumerate(folds):
        tr = labeled[labeled.index.isin(train_t)]
        te = labeled[labeled.index.isin(test_t)]
        fit_t, val_t = inner_split(train_t, val_frac, purge)
        fit, val = tr[tr.index.isin(fit_t)], tr[tr.index.isin(val_t)]
        if verbose:
            print(f"  bloque {i + 1}/{len(folds)}: {len(tr)} filas de entrenamiento ({len(fit)} + {len(val)} "
                  f"de validación), {len(te)} de prueba")

        # 1. Etapas: las de la validación salen de un HMM que solo ha visto el ajuste.
        stage_inner, stage_full = dict(fixed_labels), dict(fixed_labels)
        if "hmm" in sources:
            stage_inner["hmm"] = StageHMM(n_states, threshold).fit(fit).labels(work).to_numpy(object)
            h_full = StageHMM(n_states, threshold).fit(tr)
            stage_full["hmm"] = h_full.labels(work).to_numpy(object)
            hmm_oos[te["_pos"].to_numpy()] = stage_full["hmm"][te["_pos"].to_numpy()]
            hmm_rows.append(h_full.describe().rename_axis("estado").reset_index().assign(bloque=i))

        # 2. Elección en validación.
        y_fit, y_val = fit[target].to_numpy(), val[target].to_numpy()
        prior = float(np.clip(y_fit.mean(), 1e-3, 1 - 1e-3))
        val_general = {m: fit_predict(m, fit, val) for m in general_models}
        brier_all = {m: _brier(p, y_val) for m, p in val_general.items()}
        best_general = min(brier_all, key=brier_all.get)
        winners = {}
        for src in sources:
            lab_fit = stage_inner[src][fit["_pos"].to_numpy()]
            lab_val = stage_inner[src][val["_pos"].to_numpy()]
            for s in _stage_names(lab_fit, lab_val):
                mf, mv = lab_fit == s, lab_val == s
                n_val = int(mv.sum())
                trainable = _trainable(y_fit[mf], min_train)
                cand = {f"{m}@{GENERAL}": val_general[m][mv] for m in general_models}
                if trainable and n_val:
                    cand.update({f"{m}@{s}": fit_predict(m, fit[mf], val[mv]) for m in stage_models})
                briers = {c: _brier(p, y_val[mv]) for c, p in cand.items()} if n_val else {}
                b_prior = _brier(np.full(n_val, prior), y_val[mv]) if n_val else float("nan")
                for c, b in briers.items():
                    val_rows.append({"bloque": i, "fuente": src, "etapa": s, "candidato": c, "n": n_val,
                                     "brier": b, "skill": 1 - b / b_prior})
                if trainable and n_val >= min_val:
                    winner, why = min(briers, key=briers.get), "validación"
                else:
                    winner, why = f"{best_general}@{GENERAL}", "pocas velas: general"
                winners[(src, s)] = winner
                choice_rows.append({"bloque": i, "fuente": src, "etapa": s, "ganador": winner, "motivo": why,
                                    "n_entrenamiento": int(mf.sum()), "n_validacion": n_val})

        # 3. Reentrenar con todo el entrenamiento y predecir la prueba.
        frame = te[[target, ret_col] + keep].rename(columns={target: "up", ret_col: "fwd_ret"}).assign(bloque=i)
        y_te = frame["up"].to_numpy()
        te_general = {m: fit_predict(m, tr, te) for m in general_models}
        for b in BASELINES:
            oos_parts.setdefault(b, []).append(frame.assign(p=fit_predict(b, tr, te)))
        base = oos_parts["siempre_sube"][-1]["p"].to_numpy()
        for m in general_models:
            oos_parts.setdefault(m, []).append(frame.assign(p=te_general[m]))
        oos_parts.setdefault("general_validado", []).append(frame.assign(p=te_general[best_general]))
        for src in sources:
            lab_tr = stage_full[src][tr["_pos"].to_numpy()]
            lab_te = stage_full[src][te["_pos"].to_numpy()]
            p = te_general[best_general].copy()
            used = np.full(len(te), f"{best_general}@{GENERAL}", dtype=object)
            for s in _stage_names(lab_te):
                mt, mtr = lab_te == s, lab_tr == s
                preds = {f"{m}@{GENERAL}": te_general[m][mt] for m in general_models}
                if _trainable(tr[target].to_numpy()[mtr], min_train):
                    preds.update({f"{m}@{s}": fit_predict(m, tr[mtr], te[mt]) for m in stage_models})
                for c, pc in preds.items():
                    cand_parts.append(pd.DataFrame({"fuente": src, "etapa": s, "candidato": c, "bloque": i,
                                                    "p": pc, "up": y_te[mt], "p_base": base[mt]},
                                                   index=te.index[mt]))
                w = winners.get((src, s), f"{best_general}@{GENERAL}")
                if w in preds:
                    p[mt], used[mt] = preds[w], w
            oos_parts.setdefault(f"por_etapas_{src}", []).append(
                frame.assign(p=p, etapa=lab_te, modelo_usado=used))

    oos = {name: pd.concat(parts) for name, parts in oos_parts.items()}
    choices = pd.DataFrame(choice_rows)
    val_rank = pd.DataFrame(val_rows)
    cand = pd.concat(cand_parts) if cand_parts else pd.DataFrame()
    singles = list(general_models)
    se_h = 1 if target.startswith("up_v") else purge
    metrics = _metrics(oos, singles, [f"por_etapas_{s}" for s in sources], se_h)
    best_single = metrics.loc[singles, "skill_vs_siempre_sube"].idxmax()
    comparison = _comparison(oos, sources, best_single)
    oos_rank = _oos_rank(cand)
    agree = _agreement(choices, val_rank, oos_rank)
    share = pd.DataFrame({f"por_etapas_{s}": oos[f"por_etapas_{s}"]["etapa"].fillna("sin etapa")
                         .value_counts(normalize=True) for s in sources}).fillna(0.0)
    hmm_states = pd.concat(hmm_rows, ignore_index=True) if hmm_rows else pd.DataFrame()
    hmm_vs_rules, match = pd.DataFrame(), float("nan")
    if "hmm" in sources and rules is not None:
        tested = labeled["_pos"].to_numpy()
        tested = tested[pd.notna(hmm_oos[tested])]
        h, r = pd.Series(hmm_oos[tested]), pd.Series(rules[tested])
        hmm_vs_rules, match = compare_with_rules(h, r), agreement(h, r)
    meta = {"objetivo": target, "purga": purge, "bloques": len(folds), "modelos": list(models),
            "solo_general": [m for m in general_models if m not in stage_models], "fuentes": list(sources),
            "mejor_unico": best_single, "min_entrenamiento": min_train, "min_validacion": min_val,
            "estados_hmm": n_states, "umbral_hmm": threshold, "acuerdo_hmm_reglas": match,
            "desde": str(oos["siempre_sube"].index.min()), "hasta": str(oos["siempre_sube"].index.max())}
    return StageReport(target, metrics, comparison, choices, val_rank, oos_rank, agree, share, hmm_states,
                       hmm_vs_rules, oos, meta)


def _metrics(oos: dict, singles: list[str], staged: list[str], se_h: int) -> pd.DataFrame:
    names = list(BASELINES) + singles + ["general_validado"] + staged
    tried = singles + ["general_validado"] + staged
    base = oos["siempre_sube"]
    base_brier = scores(base, se_h)["brier"]
    base_fold = ((base["p"] - base["up"]) ** 2).groupby(base["bloque"]).mean()
    rows = {}
    for m in names:
        o = oos[m]
        s = scores(o, se_h)
        fb = ((o["p"] - o["up"]) ** 2).groupby(o["bloque"]).mean()
        s["skill_vs_siempre_sube"] = 1 - s["brier"] / base_brier
        s["bloques_ganados"] = int((fb < base_fold.reindex(fb.index)).sum()) if m != "siempre_sube" else 0
        s["bloques"] = int(len(fb))
        rows[m] = s
    t = pd.DataFrame.from_dict(rows, orient="index")
    t["p_auc_corregido"] = holm(t.loc[tried, "p_auc"]).reindex(t.index)
    t["ventaja"] = (t["p_auc_corregido"] < 0.05) & (t["skill_vs_siempre_sube"] > 0) & \
                   (t["bloques_ganados"] >= np.ceil(t["bloques"] * 2 / 3))
    return t


def _comparison(oos: dict, sources, best_single: str) -> pd.DataFrame:
    """Test pareado de Brier: ¿el modelo por etapas predice mejor que su rival en las mismas velas?"""
    rows = []
    for src in sources:
        e = oos[f"por_etapas_{src}"]
        for label, rival in (("general_validado", "general_validado"), ("mejor_unico_a_posteriori", best_single)):
            r = oos[rival]
            d = (r["p"].to_numpy() - r["up"].to_numpy()) ** 2 - (e["p"].to_numpy() - e["up"].to_numpy()) ** 2
            se = se_by_day(d, e.index)
            z = d.mean() / se if se and se == se else float("nan")
            per_fold = pd.Series(d).groupby(e["bloque"].to_numpy()).mean()
            rows.append({"fuente": src, "rival": label, "modelo_rival": rival, "n": len(d),
                         "brier_etapas": _brier(e["p"], e["up"]), "brier_rival": _brier(r["p"], r["up"]),
                         "mejora_brier": float(d.mean()), "z": float(z),
                         "p_valor": float(math.erfc(abs(z) / math.sqrt(2))) if z == z else float("nan"),
                         "bloques_ganados": int((per_fold > 0).sum()), "bloques": int(len(per_fold))})
    return pd.DataFrame(rows)


def _oos_rank(cand: pd.DataFrame) -> pd.DataFrame:
    """Cada candidato en las velas de prueba de su etapa (todos los bloques juntos)."""
    if cand.empty:
        return pd.DataFrame()
    rows = []
    for (src, s, c), g in cand.groupby(["fuente", "etapa", "candidato"], sort=False):
        y, p = g["up"].to_numpy(), g["p"].to_numpy()
        b, b0 = _brier(p, y), _brier(g["p_base"], y)
        rows.append({"fuente": src, "etapa": s, "candidato": c, "n": len(g), "bloques": g["bloque"].nunique(),
                     "brier": b, "skill": 1 - b / b0,
                     "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")})
    out = pd.DataFrame(rows)
    out["puesto"] = out.groupby(["fuente", "etapa"])["skill"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["fuente", "etapa", "puesto"]).reset_index(drop=True)


def _agreement(choices: pd.DataFrame, val_rank: pd.DataFrame, oos_rank: pd.DataFrame) -> pd.DataFrame:
    """¿Coinciden la validación y la prueba? Correlación de rangos y puesto real del más elegido."""
    if choices.empty or oos_rank.empty:
        return pd.DataFrame()
    val_mean = val_rank.groupby(["fuente", "etapa", "candidato"])["skill"].mean() if len(val_rank) else \
        pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["fuente", "etapa", "candidato"]))
    rows = []
    for (src, s), g in choices.groupby(["fuente", "etapa"]):
        chosen = g.loc[g["motivo"] == "validación", "ganador"]
        o = oos_rank[(oos_rank["fuente"] == src) & (oos_rank["etapa"] == s)].set_index("candidato")
        v = val_mean.loc[(src, s)] if (src, s) in val_mean.index.droplevel(2) else pd.Series(dtype=float)
        common = v.index.intersection(o.index)
        rho = float(v[common].rank().corr(o.loc[common, "skill"].rank())) if len(common) >= 3 else float("nan")
        top = chosen.mode().iloc[0] if len(chosen) else None
        rank = int(o.loc[top, "puesto"]) if top in o.index else None
        rows.append({"fuente": src, "etapa": s, "bloques_con_eleccion": len(chosen), "bloques": len(g),
                     "mas_elegido": top, "veces_elegido": int((chosen == top).sum()) if top else 0,
                     "puesto_fuera_de_muestra": rank, "mejor_fuera_de_muestra": o["skill"].idxmax() if len(o) else None,
                     "correlacion_rangos": rho,
                     "estabilidad": "sin datos" if rho != rho else
                     "estable" if rho >= 0.5 else "dudosa" if rho > 0 else "ruidosa"})
    return pd.DataFrame(rows)
