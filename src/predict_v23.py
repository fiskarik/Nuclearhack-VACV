"""Reproduce VACV v23 predictions using SMILES only.

Usage:
python predict_v23.py --state v23_model_state.json --test test.csv \
    --sample sample_submit.csv --output submission.csv

The competition concentration fields are never read. No target is read from test.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vacv_core as V
import source_model as M


def predict(state_file, test_file, sample_file, output_file):
    state = json.loads(Path(state_file).read_text(encoding="utf-8"))
    test = pd.read_csv(test_file, usecols=["index", "SMILES"])
    sample = pd.read_csv(sample_file, usecols=["index"])
    if not (test["index"].is_unique and sample["index"].is_unique and
            set(test["index"]) == set(sample["index"])):
        raise ValueError("Sample/test indices do not match")
    smiles = test["SMILES"].tolist()
    with threadpool_limits(limits=2):
        V.GP_KERNELS = tuple(state["kernels_old"])
        V.GP_ALPHAS = tuple(state["alphas_old"])
        old = M.predict(state["old"], smiles)
        V.GP_KERNELS = tuple(state["kernels_new"])
        V.GP_ALPHAS = tuple(state["alphas_new"])
        new = M.predict(state["new"], smiles)
    w = float(state["weight_new"])
    pred = (1.0 - w) * old + w * new
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise RuntimeError("Invalid predictions")
    aligned = pd.Series(pred, index=test["index"])
    out = sample.copy()
    out["SI VACV"] = out["index"].map(aligned)
    if out["SI VACV"].isna().any():
        raise RuntimeError("Prediction alignment failed")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved {len(out)} predictions: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict(args.state, args.test, args.sample, args.output)
