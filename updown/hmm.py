"""Etapas con probabilidad: modelo oculto de Markov (HMM) gaussiano, sin mirar al futuro.

Las etapas de `regimes.py` salen de reglas fijas. Aquí se aprende de los propios
datos que el mercado pasa por unos pocos estados ocultos (4 por defecto), cada
uno con su volatilidad y su tendencia típicas, y con probabilidades de pasar de
uno a otro. El resultado no es una etiqueta sino probabilidades, del tipo «75 %
calma, 20 % alcista, 5 % crisis».

Observación de cada vela (las mismas magnitudes que usan las reglas, sin escala
para poder juntar activos):

  · hmm_vol  : volatilidad del último día, en desviaciones típicas respecto a su
               último año (z-score de su logaritmo, con el año ANTERIOR a la vela).
  · hmm_tend : retorno de los últimos ~30 días dividido por la volatilidad de ese
               plazo. Es la cantidad que las reglas comparan con 1 para decidir si
               hay tendencia clara.

Cómo se evita mirar al futuro:

  · Los parámetros (medias, covarianzas, transiciones) se ajustan con EM SOLO con
    el periodo de entrenamiento de cada bloque.
  · La probabilidad de cada vela es la FILTRADA, P(estado_t | observaciones hasta
    t): el paso hacia delante del algoritmo forward. Nunca la suavizada (forward-
    backward) ni la secuencia de Viterbi, que se recalculan con lo que vino
    después. El paso hacia atrás solo se usa dentro de EM, con datos de entrenamiento.

Los números de estado de un HMM son arbitrarios (cambian de un ajuste a otro),
así que cada estado se nombra por la zona en la que caen su volatilidad y su
tendencia medias, con límites fijados de antemano como en las reglas:

  · volatilidad > +0,5 : «crisis» si la tendencia es negativa, «volatil» si no.
  · volatilidad < -0,5 y sin tendencia clara (|tendencia| < 1): «calma».
  · el resto, por su tendencia: «alcista» (> +0,5), «bajista» (< -0,5) o «lateral».

Dos estados en la misma zona comparten nombre y su probabilidad se suma: la etapa
es la zona, no el estado. La etapa solo se da por buena si su probabilidad supera
el umbral (90 % por defecto); si no, la vela es «incierta».

Ojo: esa probabilidad es la del propio modelo, no una frecuencia comprobada. Un
estado oculto no tiene verdad con la que contrastarlo; lo que sí se puede medir es
si predecir por etapas mejora fuera de muestra (`stages.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import bars_per_day, regime_inputs

HMM_COLUMNS = ("hmm_vol", "hmm_tend")
UNCERTAIN = "incierta"
STATE_LABELS = {
    "calma": "Calma",
    "alcista": "Alcista",
    "bajista": "Bajista",
    "lateral": "Lateral",
    "crisis": "Crisis (volátil y cayendo)",
    "volatil": "Volátil sin caída",
    UNCERTAIN: "Incierta (< umbral)",
}


def hmm_observations(bars: pd.DataFrame, inputs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Observaciones del HMM para cada vela, calculadas solo con el pasado."""
    x = regime_inputs(bars) if inputs is None else inputs
    bpd = bars_per_day(bars.index)
    month = max(int(round(30 * bpd)), 20)
    year = max(int(round(365 * bpd)), 250)
    logvol = np.log(x["vol"].where(x["vol"] > 0))
    mu = logvol.rolling(year, min_periods=month).mean().shift(1)
    sd = logvol.rolling(year, min_periods=month).std().shift(1)
    out = pd.DataFrame(index=bars.index)
    out["hmm_vol"] = ((logvol - mu) / sd).clip(-5, 5)
    out["hmm_tend"] = (x["tendencia"] / (x["vol"] * np.sqrt(month))).clip(-5, 5)
    return out.replace([np.inf, -np.inf], np.nan)


class GaussianHMM:
    """HMM con emisiones gaussianas de covarianza completa, ajustado por EM (Baum-Welch).

    Las secuencias (una por activo) se procesan a la vez, rellenadas al final con
    observaciones vacías: una observación NaN no aporta información (verosimilitud 1
    en todos los estados), así que el relleno no cambia ni el filtro ni EM.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 100, tol: float = 1e-6, reg: float = 1e-3,
                 seed: int = 0):
        self.n_states, self.n_iter, self.tol, self.reg, self.seed = n_states, n_iter, tol, reg, seed

    # -- utilidades ---------------------------------------------------------------
    @staticmethod
    def _pad(seqs: list[np.ndarray]) -> np.ndarray:
        d = seqs[0].shape[1]
        out = np.full((len(seqs), max(len(s) for s in seqs), d), np.nan)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return out

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log N(x | media_k, cov_k) con forma (S, T, K); 0 donde la observación falta."""
        S, T, D = X.shape
        flat = X.reshape(-1, D)
        ok = ~np.isnan(flat).any(axis=1)
        out = np.zeros((len(flat), self.n_states))
        for k in range(self.n_states):
            chol = np.linalg.cholesky(self.covars_[k])
            z = np.linalg.solve(chol, (flat[ok] - self.means_[k]).T)
            logdet = 2 * np.log(np.diag(chol)).sum()
            out[ok, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + (z ** 2).sum(axis=0))
        return out.reshape(S, T, self.n_states)

    def _forward(self, logb: np.ndarray):
        """Paso hacia delante normalizado: alpha[s, t] = P(estado_t | obs hasta t)."""
        S, T, K = logb.shape
        top = logb.max(axis=2, keepdims=True)
        b = np.exp(logb - top)
        alpha = np.empty((S, T, K))
        scale = np.empty((S, T))
        a = self.startprob_ * b[:, 0]
        scale[:, 0] = a.sum(axis=1)
        alpha[:, 0] = a / scale[:, :1]
        for t in range(1, T):
            a = (alpha[:, t - 1] @ self.transmat_) * b[:, t]
            scale[:, t] = a.sum(axis=1)
            alpha[:, t] = a / scale[:, t:t + 1]
        loglik = float((np.log(scale) + top[..., 0]).sum())
        return alpha, scale, b, loglik

    # -- ajuste -------------------------------------------------------------------
    def _init(self, X: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        obs = X.reshape(-1, X.shape[2])
        obs = obs[~np.isnan(obs).any(axis=1)]
        rng = np.random.default_rng(self.seed)
        sample = obs[rng.choice(len(obs), min(len(obs), 20000), replace=False)]
        km = KMeans(self.n_states, n_init=10, random_state=self.seed).fit(sample)
        K, D = self.n_states, obs.shape[1]
        self.means_ = km.cluster_centers_
        self.covars_ = np.stack([np.cov(sample[km.labels_ == k].T) + self.reg * np.eye(D)
                                 if (km.labels_ == k).sum() > D else np.cov(sample.T) for k in range(K)])
        self.transmat_ = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(self.transmat_, 0.95)
        self.startprob_ = np.full(K, 1.0 / K)

    def fit(self, seqs: list[np.ndarray]) -> "GaussianHMM":
        X = self._pad([np.asarray(s, dtype=float) for s in seqs])
        S, T, D = X.shape
        K = self.n_states
        lengths = np.array([len(s) for s in seqs])
        real = np.arange(T)[None, :] < lengths[:, None]             # (S, T): paso real, no relleno
        seen = real & ~np.isnan(X).any(axis=2)                      # paso con observación
        Xz = np.where(seen[..., None], X, 0.0)
        n_obs = max(int(seen.sum()), 1)
        self._init(X)
        self.loglik_history_ = []
        for _ in range(self.n_iter):
            alpha, scale, b, loglik = self._forward(self._log_emission(X))
            self.loglik_history_.append(loglik)
            beta = np.empty_like(alpha)
            beta[:, -1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[:, t] = ((b[:, t + 1] * beta[:, t + 1]) @ self.transmat_.T) / scale[:, t + 1:t + 2]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=2, keepdims=True)
            # Transiciones esperadas: xi_t(i, j) ∝ alpha_{t-1}(i) A_ij b_t(j) beta_t(j), solo entre pasos reales.
            right = (b[:, 1:] * beta[:, 1:]) / scale[:, 1:, None] * real[:, 1:, None]
            xi = np.einsum("sti,stj->ij", alpha[:, :-1], right) * self.transmat_
            self.startprob_ = np.clip(gamma[:, 0].mean(axis=0), 1e-6, None)
            self.startprob_ /= self.startprob_.sum()
            self.transmat_ = np.clip(xi / xi.sum(axis=1, keepdims=True), 1e-6, None)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            w = gamma * seen[..., None]                                # (S, T, K)
            wk = w.sum(axis=(0, 1))
            self.means_ = np.einsum("stk,std->kd", w, Xz) / wk[:, None]
            for k in range(K):
                dz = (Xz - self.means_[k]) * np.sqrt(w[..., k:k + 1])
                self.covars_[k] = np.einsum("std,ste->de", dz, dz) / wk[k] + self.reg * np.eye(D)
            if len(self.loglik_history_) > 1 and \
                    abs(self.loglik_history_[-1] - self.loglik_history_[-2]) < self.tol * n_obs:
                break
        return self

    # -- uso ----------------------------------------------------------------------
    def filter(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        """Probabilidades filtradas P(estado_t | obs_1..t) de cada secuencia, (T, K) cada una."""
        X = self._pad([np.asarray(s, dtype=float) for s in seqs])
        alpha = self._forward(self._log_emission(X))[0]
        return [alpha[i, :len(s)] for i, s in enumerate(seqs)]

    def durations(self) -> np.ndarray:
        """Duración media de cada estado en velas: 1 / (1 - P(quedarse)); inf si nunca se salió de él."""
        leave = 1.0 - np.diag(self.transmat_)
        return np.where(leave < 1e-4, np.inf, 1.0 / np.maximum(leave, 1e-12))


def name_states(means: np.ndarray, level: float = 0.5) -> list[str]:
    """Nombre de cada estado según la zona de su volatilidad (columna 0) y tendencia (columna 1) medias."""
    names = []
    for vol, trend in means[:, :2]:
        if vol > level:
            names.append("crisis" if trend < 0 else "volatil")
        elif vol < -level and abs(trend) < 1:
            names.append("calma")
        elif trend > level:
            names.append("alcista")
        elif trend < -level:
            names.append("bajista")
        else:
            names.append("lateral")
    return names


def _sequences(rows: pd.DataFrame) -> list[np.ndarray]:
    """Posiciones de las filas de cada activo, en orden temporal."""
    pos = np.arange(len(rows))
    if "symbol" not in rows:
        return [pos[np.argsort(rows.index.to_numpy(), kind="stable")]]
    out = []
    for _, g in pd.Series(pos, index=rows.index).groupby(rows["symbol"].to_numpy()):
        out.append(g.to_numpy()[np.argsort(g.index.to_numpy(), kind="stable")])
    return out


class StageHMM:
    """HMM sobre (hmm_vol, hmm_tend) con estados con nombre y probabilidades filtradas.

    `fit` recibe SOLO las filas de entrenamiento. `proba` filtra cualquier tabla con
    los parámetros ya ajustados: la fila t solo depende de las observaciones de su
    activo hasta t.
    """

    def __init__(self, n_states: int = 4, threshold: float = 0.9, seed: int = 0):
        self.n_states, self.threshold, self.seed = n_states, threshold, seed

    def fit(self, rows: pd.DataFrame) -> "StageHMM":
        X = rows[list(HMM_COLUMNS)].to_numpy(float)
        seqs = []
        for idx in _sequences(rows):
            s = X[idx]
            ok = ~np.isnan(s).any(axis=1)
            if ok.any():                                    # sin el calentamiento inicial (NaN)
                seqs.append(s[np.argmax(ok):])
        if sum((~np.isnan(q).any(axis=1)).sum() for q in seqs) < 50 * self.n_states:
            raise ValueError("Muy pocas velas con observación para ajustar el HMM (hace falta ~1 mes de "
                             "calentamiento antes del entrenamiento).")
        self.hmm_ = GaussianHMM(self.n_states, seed=self.seed).fit(seqs)
        self.names_ = name_states(self.hmm_.means_)        # uno por estado (puede repetirse)
        self.stages_ = sorted(set(self.names_))
        return self

    def proba(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Probabilidad filtrada de cada etapa (suma de sus estados), una columna por etapa."""
        X = rows[list(HMM_COLUMNS)].to_numpy(float)
        idxs = _sequences(rows)
        P = np.full((len(rows), self.n_states), np.nan)
        for idx, alpha in zip(idxs, self.hmm_.filter([X[i] for i in idxs])):
            P[idx] = alpha
        P[np.isnan(X).any(axis=1)] = np.nan                # sin observación no hay etapa
        names = np.array(self.names_)
        stages = np.column_stack([P[:, names == s].sum(axis=1) for s in self.stages_])
        stages[np.isnan(P).any(axis=1)] = np.nan
        return pd.DataFrame(stages, index=rows.index, columns=self.stages_)

    def labels(self, rows: pd.DataFrame, proba: pd.DataFrame | None = None) -> pd.Series:
        """Etapa con probabilidad >= umbral; si no, «incierta». NaN si aún no hay observación."""
        P = self.proba(rows) if proba is None else proba
        values = P.to_numpy()
        ok = ~np.isnan(values).any(axis=1)
        filled = np.where(ok[:, None], values, -1.0)
        best, conf = filled.argmax(axis=1), filled.max(axis=1)
        out = np.full(len(P), np.nan, dtype=object)
        names = np.array(list(P.columns), dtype=object)
        out[ok] = np.where(conf[ok] >= self.threshold, names[best[ok]], UNCERTAIN)
        return pd.Series(out, index=P.index, name="etapa_hmm")

    def describe(self) -> pd.DataFrame:
        """Características medias de cada estado y cuánto dura de media (en velas)."""
        return pd.DataFrame({"volatilidad_z": self.hmm_.means_[:, 0], "tendencia_z": self.hmm_.means_[:, 1],
                             "duracion_velas": self.hmm_.durations()}, index=self.names_)

    def transitions(self) -> pd.DataFrame:
        return pd.DataFrame(self.hmm_.transmat_, index=self.names_, columns=self.names_)


def compare_with_rules(hmm_labels: pd.Series, rule_labels: pd.Series) -> pd.DataFrame:
    """Tabla cruzada: de las velas de cada etapa del HMM, qué parte cae en cada régimen de reglas."""
    both = pd.DataFrame({"hmm": hmm_labels, "reglas": rule_labels}).dropna()
    return pd.crosstab(both["hmm"], both["reglas"], normalize="index")


def agreement(hmm_labels: pd.Series, rule_labels: pd.Series) -> float:
    """Parte de las velas con etapa segura en las que HMM y reglas dicen lo mismo (mismos nombres)."""
    both = pd.DataFrame({"hmm": hmm_labels, "reglas": rule_labels}).dropna()
    both = both[both["hmm"] != UNCERTAIN]
    return float((both["hmm"] == both["reglas"]).mean()) if len(both) else float("nan")
