"""Transformer sobre la ventana de barras recientes (necesita PyTorch).

Entrada por fila: una secuencia de `seq_len` barras con 3-4 canales (retorno,
rango, cuerpo, volumen relativo) y, aparte, las variables tabulares de siempre.

    barras (L x C) -> lineal a d_model + posición aprendida -> 2 capas encoder
                   -> media temporal ─┐
    tabulares ────────────────────────┴-> MLP -> logit P(sube)

Con `y` de varias columnas (DataFrame) la red da una probabilidad por columna y
se entrena con todas a la vez: así predice cada una de las próximas velas.

Entrenamiento: AdamW, entropía cruzada binaria, parada temprana con el último
15% del entrenamiento (en orden temporal, nunca mezclado). Semilla fija. Usa la
GPU si la hay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("El modelo 'transformer' necesita PyTorch: pip install torch") from exc
    return torch


class TransformerModel:
    name = "transformer"
    uses_sequence = True

    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 epochs: int = 25, batch: int = 512, lr: float = 1e-3, patience: int = 3, seed: int = 0):
        self.d_model, self.layers, self.heads, self.dropout = d_model, layers, heads, dropout
        self.epochs, self.batch, self.lr, self.patience, self.seed = epochs, batch, lr, patience, seed

    # -- forma de los datos ---------------------------------------------------
    def _split(self, X: pd.DataFrame):
        seq_cols = [c for c in X.columns if c.startswith("seq_")]
        tab_cols = [c for c in X.columns if not c.startswith("seq_")]
        channels = sorted({c.split("_")[1] for c in seq_cols}, key=lambda ch: ("r", "hl", "body", "vz").index(ch))
        lags = sorted({int(c.rsplit("_", 1)[1]) for c in seq_cols}, reverse=True)  # del más antiguo al actual
        seq = np.stack([X[[f"seq_{ch}_{lag}" for lag in lags]].to_numpy(np.float32) for ch in channels], axis=-1)
        return seq, X[tab_cols].to_numpy(np.float32)

    def _net(self, n_channels: int, seq_len: int, n_tab: int, n_out: int = 1):
        torch = _torch()
        nn = torch.nn
        d = self.d_model

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(n_channels, d)
                s.pos = nn.Parameter(torch.zeros(1, seq_len, d))
                layer = nn.TransformerEncoderLayer(d, self.heads, dim_feedforward=2 * d, dropout=self.dropout,
                                                   batch_first=True, norm_first=True)
                s.enc = nn.TransformerEncoder(layer, self.layers)
                s.tab = nn.Sequential(nn.Linear(n_tab, d), nn.GELU()) if n_tab else None
                s.head = nn.Sequential(nn.LayerNorm(d * (2 if n_tab else 1)), nn.Dropout(self.dropout),
                                       nn.Linear(d * (2 if n_tab else 1), d), nn.GELU(), nn.Linear(d, n_out))

            def forward(s, seq, tab):
                z = s.enc(s.inp(seq) + s.pos).mean(dim=1)
                if s.tab is not None:
                    z = torch.cat([z, s.tab(tab)], dim=1)
                return s.head(z)

        return Net()

    # -- entrenamiento ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransformerModel":
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        seq, tab = self._split(X)
        ya = np.asarray(y, dtype=np.float32)
        self.multi_ = ya.ndim == 2
        ya = ya.reshape(len(ya), -1)  # una columna por objetivo (p. ej. vela 1..4 y hora)
        self.prior_ = ya.mean(axis=0)
        if any(len(np.unique(ya[:, j])) < 2 for j in range(ya.shape[1])):
            self.net_ = None
            return self
        # Normalización con estadísticas del entrenamiento (por canal y por variable).
        self.seq_mu_, self.seq_sd_ = seq.mean(axis=(0, 1)), seq.std(axis=(0, 1)) + 1e-8
        self.tab_mu_, self.tab_sd_ = tab.mean(axis=0), tab.std(axis=0) + 1e-8
        seq, tab = self._norm(seq, tab)

        cut = int(len(ya) * 0.85)
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device_)
        tr = (T(seq[:cut]), T(tab[:cut]), T(ya[:cut]))
        va = (T(seq[cut:]), T(tab[cut:]), T(ya[cut:]))

        net = self._net(seq.shape[2], seq.shape[1], tab.shape[1], ya.shape[1]).to(self.device_)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-2)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = float("inf"), None, 0
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            order = torch.randperm(cut, generator=gen).to(self.device_)
            for i in range(0, cut, self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(net(tr[0][idx], tr[1][idx]), tr[2][idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            val = self._loss(net, va, loss_fn)
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.eval()
        return self

    def _norm(self, seq, tab):
        return (seq - self.seq_mu_) / self.seq_sd_, (tab - self.tab_mu_) / self.tab_sd_

    def _loss(self, net, data, loss_fn) -> float:
        torch = _torch()
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(data[2]), 4096):
                s = slice(i, i + 4096)
                total += float(loss_fn(net(data[0][s], data[1][s]), data[2][s])) * len(data[2][s])
                n += len(data[2][s])
        return total / max(n, 1)

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.net_ is None:
            out = np.tile(self.prior_, (len(X), 1))
            return out if self.multi_ else out[:, 0]
        torch = _torch()
        seq, tab = self._norm(*self._split(X))
        out = []
        with torch.no_grad():
            for i in range(0, len(seq), 4096):
                s = torch.from_numpy(np.ascontiguousarray(seq[i:i + 4096])).to(self.device_)
                t = torch.from_numpy(np.ascontiguousarray(tab[i:i + 4096])).to(self.device_)
                out.append(torch.sigmoid(self.net_(s, t)).cpu().numpy())
        out = np.concatenate(out).astype(float)
        return out if self.multi_ else out[:, 0]
