#!/usr/bin/env python3
"""
promotion_gate.py — Step 4 of the model-improvement pipeline.

ONE gate, ONE model. Champion/challenger evaluated walk-forward on identical
features/labels/horizon; the challenger is promoted ONLY if it beats the
champion consistently across time folds — not on one lucky split.

Walk-forward design (fixes the narrow single-split weakness):
  * Timeline is cut into N_FOLDS contiguous time blocks.
  * For fold k (k = 1..N_FOLDS-1): the GBT challenger is RETRAINED on all
    data strictly before fold k (expanding window), then both models are
    scored on fold k only. The champion transformer is evaluated zero-shot
    (it is a fixed artifact; retraining it is out of scope for the gate).
  * No fold ever sees data from its own future, in either model's path.

Honesty caveats printed with results:
  * If the champion's original training window overlaps early folds, its
    early-fold scores are optimistic. Judge the LATER folds hardest.
  * Overlapping-window forward returns inflate significance; ICs are a
    ranking signal across folds, not a p-value contest.

Promotion rule (primary metric = Spearman IC vs forward return):
    promote iff challenger wins >= ceil(win_frac * n_scored_folds) folds
    AND mean challenger IC > mean champion IC across those folds.

By default this script is REPORT-ONLY. With --apply it additionally copies
the challenger artifact to champion_promoted.joblib and prints the exact
MODEL_PATH change to deploy (hot-reload picks it up without a restart).
It NEVER edits config.py or restarts anything itself.

Usage:
    python promotion_gate.py                     # report only
    python promotion_gate.py --folds 6 --apply   # also stage promotion
    python promotion_gate.py --days 120          # shorter history
"""

import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import SafeMLPredictor
from config import MODEL_PATH, SEQUENCE_LEN, GBT_CHALLENGER_PATH

N_FOLDS_DEFAULT = 5
WIN_FRAC = 0.6


# ── pure decision logic (unit-tested without any data/network) ────────────────
def decide_promotion(challenger_ics: list, champion_ics: list,
                     win_frac: float = WIN_FRAC) -> tuple:
    """Return (promote?, reason). Both inputs are per-fold Spearman IC lists
    (None entries allowed for degenerate folds and are dropped)."""
    pairs = [(c, h) for c, h in zip(challenger_ics, champion_ics)
             if c is not None and h is not None]
    if not pairs:
        return False, "no scoreable folds"
    c_wins = sum(1 for c, h in pairs if c > h)
    need = math.ceil(win_frac * len(pairs))
    c_mean = float(np.mean([c for c, _ in pairs]))
    h_mean = float(np.mean([h for _, h in pairs]))
    reason = (f"challenger won {c_wins}/{len(pairs)} folds "
              f"(needed {need}); meanIC {c_mean:+.4f} vs {h_mean:+.4f}")
    return (c_wins >= need and c_mean > h_mean), reason


def fold_boundaries(ts: pd.Series, n_folds: int) -> list:
    """Contiguous time-fold edges: returns list of (start_ts, end_ts) tuples
    (values preserved as the series' native dtype, e.g. pd.Timestamp)."""
    uniq = np.sort(pd.unique(ts))
    edges = np.array_split(uniq, n_folds)
    return [(ts.iloc[0].__class__(e[0]), ts.iloc[0].__class__(e[-1]))
            for e in edges if len(e)]


def evaluate_fold_metrics(y_true: np.ndarray, proba: np.ndarray,
                          fwd_ret: np.ndarray):
    """AUC + Spearman IC + directional-edge proxy for one fold."""
    if len(y_true) < 100:
        return None
    from scipy.stats import spearmanr
    m = {"n": int(len(y_true))}
    try:
        from sklearn.metrics import roc_auc_score
        m["auc"] = (float(roc_auc_score(y_true, proba))
                    if len(np.unique(y_true)) > 1 else None)
    except Exception:
        m["auc"] = None
    ic_r, _ = spearmanr(proba, fwd_ret)
    m["ic"] = None if (ic_r is None or np.isnan(ic_r)) else float(ic_r)
    hi = proba >= 0.55
    lo = proba <= 0.45
    m["edge"] = (float(fwd_ret[hi].mean() - fwd_ret[lo].mean())
                 if hi.sum() >= 10 and lo.sum() >= 10 else None)
    return m


# ── model scoring helpers ──────────────────────────────────────────────────────
def score_champion_rows(predictor: SafeMLPredictor, feats_by_sym: dict,
                        rows: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Zero-shot champion probabilities for eval rows.

    rows must carry columns [symbol, pos] where pos indexes that symbol's
    feature frame; builds seq_len windows ending at each pos.
    """
    import torch

    probs = np.full(len(rows), 0.5, dtype=np.float64)
    order, tensors = [], []
    scaler = predictor.scaler
    for r_i, (sym, pos) in enumerate(zip(rows["symbol"].values,
                                         rows["pos"].values)):
        F = feats_by_sym.get(sym)
        if F is None or pos < seq_len - 1:
            continue
        win = F[pos - seq_len + 1: pos + 1]
        x = np.nan_to_num(win.astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e6, 1e6)
        if scaler is not None:
            x = scaler.transform(x).astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        tensors.append(torch.tensor(x).unsqueeze(0))
        order.append(r_i)
    if tensors:
        batch = torch.cat(tensors, dim=0).to(predictor.device)
        with torch.no_grad():
            p = torch.sigmoid(predictor.model(batch)).squeeze(-1).cpu().numpy()
        for j, r_i in enumerate(order):
            probs[r_i] = float(p[j])
    return probs


def _fmt(m) -> str:
    if not m:
        return "n/a"
    auc = f"AUC={m['auc']:.3f}" if m.get("auc") is not None else "AUC=n/a"
    ic = f"IC={m['ic']:+.4f}" if m.get("ic") is not None else "IC=n/a"
    edge = (f"edge={m['edge'] * 100:+.3f}%"
            if m.get("edge") is not None else "edge=n/a")
    return f"{auc} {ic} {edge}"


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--folds", type=int, default=N_FOLDS_DEFAULT)
    ap.add_argument("--challenger", default=GBT_CHALLENGER_PATH)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--apply", action="store_true",
                    help="stage promotion artifacts if the gate passes")
    args = ap.parse_args()

    if not os.path.exists(args.challenger):
        raise SystemExit(f"Challenger artifact not found: {args.challenger} "
                         f"— run train_gbt_baseline.py first")
    if not os.path.exists(args.model):
        raise SystemExit(f"Champion artifact not found: {args.model}")

    # Reuse the trainer's dataset builder so features/labels are IDENTICAL.
    from train_transformer import (
        _get_data_client, fetch_bars, TRAIN_SYMBOLS, TARGET_HORIZON,
    )
    from train_gbt_baseline import build_dataset, WARMUP_BARS
    from sklearn.ensemble import HistGradientBoostingClassifier

    print(f"── promotion gate ── folds={args.folds}, horizon={TARGET_HORIZON}")
    client = _get_data_client()
    data = build_dataset(client, TRAIN_SYMBOLS, args.days)

    # Rebuild per-symbol frames so eval rows can be mapped back to contiguous
    # windows for the sequence-model champion.
    feats_by_sym, ts_index_by_sym = {}, {}
    for sym in TRAIN_SYMBOLS:
        df = fetch_bars(client, sym, args.days)
        if df is None or len(df) < SEQUENCE_LEN + TARGET_HORIZON + 50:
            continue
        feats_by_sym[sym] = add_features(df)[FEATURE_COLS].values.astype(np.float64)
        ts_index_by_sym[sym] = df.index

    pos_list = []
    for sym, ts in zip(data["symbol"].values, data["ts"].values):
        tidx = ts_index_by_sym.get(sym)
        if tidx is None:
            pos_list.append(-1)
            continue
        try:
            pos_list.append(int(tidx.get_loc(ts)))
        except KeyError:
            pos_list.append(-1)
    data["pos"] = pos_list
    data = data[data["pos"] >= WARMUP_BARS].reset_index(drop=True)

    champion = SafeMLPredictor(model_path=args.model, seq_len=SEQUENCE_LEN)

    folds = fold_boundaries(data["ts"], args.folds)
    chal_ics, champ_ics, table = [], [], []

    print("Caveat: champion evaluated zero-shot on ALL folds; if its training "
          "window overlaps early folds, judge later folds hardest.")

    for k in range(1, len(folds)):
        f_start, f_end = folds[k]
        tr_mask = data["ts"] < f_start
        ev_mask = (data["ts"] >= f_start) & (data["ts"] <= f_end)
        X_tr = data.loc[tr_mask, FEATURE_COLS].values.astype(np.float64)
        y_tr = data.loc[tr_mask, "label"].values
        ev = data.loc[ev_mask].reset_index(drop=True)
        if len(ev) < 100 or len(np.unique(y_tr)) < 2:
            print(f"Fold {k}: skipped (train={len(y_tr)}, eval={len(ev)})")
            chal_ics.append(None)
            champ_ics.append(None)
            continue

        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )
        clf.fit(X_tr, y_tr)
        chal_p = clf.predict_proba(
            ev[FEATURE_COLS].values.astype(np.float64)
        )[:, list(clf.classes_).index(1)]

        champ_p = score_champion_rows(champion, feats_by_sym, ev, SEQUENCE_LEN)

        y_ev = ev["label"].values
        fwd = ev["fwd_ret"].values
        mc = evaluate_fold_metrics(y_ev, chal_p, fwd)
        mh = evaluate_fold_metrics(y_ev, champ_p, fwd)
        chal_ics.append(mc["ic"] if mc else None)
        champ_ics.append(mh["ic"] if mh else None)
        print(f"Fold {k} [{f_start} → {f_end}] n={len(ev)}")
        print(f"   challenger : {_fmt(mc)}")
        print(f"   champion   : {_fmt(mh)}")
        table.append({"fold": k, "span": [str(f_start), str(f_end)],
                      "n": len(ev), "challenger": mc, "champion": mh})

    promote, reason = decide_promotion(chal_ics, champ_ics)
    verdict = "PROMOTE" if promote else "KEEP CHAMPION"
    print(f"GATE DECISION: {verdict} — {reason}")

    decision = {
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict, "reason": reason,
        "challenger_artifact": os.path.abspath(args.challenger),
        "champion_artifact": os.path.abspath(args.model),
        "folds": table,
    }
    with open("PROMOTION_DECISION.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, default=str)
    print("Wrote PROMOTION_DECISION.json")

    if promote and args.apply:
        staged = "champion_promoted.joblib"
        shutil.copyfile(args.challenger, staged)
        print(f"Staged {staged}. To deploy: set MODEL_PATH={staged} and restart "
              f"(or rely on hot-reload after changing the mounted file).")


if __name__ == "__main__":
    main()