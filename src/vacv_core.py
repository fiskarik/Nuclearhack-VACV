# Structural primitives extracted unchanged from user v13.1. Legacy validation and external ingestion removed.
from __future__ import annotations

import argparse, json, platform, time

from pathlib import Path

import numpy as np, pandas as pd

from scipy.stats import spearmanr

from rdkit import Chem, DataStructs, RDLogger, rdBase

from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdMolDescriptors

from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit.ML.Cluster import Butina

RDLogger.DisableLog('rdApp.*')

SMI, TGT = 'SMILES', 'SI VACV'

def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None

def scaffold(s):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or 'ACYCLIC'

def load(train_csv, test_csv, sample_csv):
    tr = pd.read_csv(train_csv, usecols=[SMI, TGT]); te = pd.read_csv(test_csv, usecols=['index', SMI])
    sm = pd.read_csv(sample_csv, usecols=['index'])           # IDs/order only - never the sample's values
    tr[TGT] = pd.to_numeric(tr[TGT], errors='coerce'); tr['c'] = tr[SMI].map(canon); te['c'] = te[SMI].map(canon)
    bad = tr.c.isna() | ~np.isfinite(tr[TGT]) | (tr[TGT] < 0)
    if bad.any(): print('dropping invalid train rows:', int(bad.sum())); tr = tr[~bad].reset_index(drop=True)
    if te.c.isna().any(): raise ValueError('invalid test SMILES')
    if not (te['index'].is_unique and sm['index'].is_unique and set(te['index']) == set(sm['index'])): raise ValueError('test/sample index mismatch')
    tr['scaf'] = tr.c.map(scaffold)
    return tr, te, sm

def _bulk(fps): return np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])

def build_kernels(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    K = {}
    for r in (2, 3):
        g = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=2048, includeChirality=True)
        K[f'mc{r}'] = _bulk([g.GetCountFingerprint(m) for m in mols])          # count (MinMax) Tanimoto
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    K['mb2'] = _bulk([g.GetFingerprint(m) for m in mols])                      # bit Tanimoto
    K['maccs'] = _bulk([MACCSkeys.GenMACCSKeys(m) for m in mols])
    K['ap'] = _bulk([rdMolDescriptors.GetHashedAtomPairFingerprint(m, 2048) for m in mols])
    K['mix'] = (K['mc2'] + K['maccs'] + K['ap']) / 3
    return K

TF = {'raw': (lambda y: y, lambda z: np.maximum(z, 0)), 'sqrt': (np.sqrt, lambda z: np.maximum(z, 0) ** 2),
      'q4': (lambda y: y ** .25, lambda z: np.maximum(z, 0) ** 4), 'log': (np.log1p, lambda z: np.expm1(np.clip(z, 0, 20)))}

def smear(tf, mu, v):
    """E[g^-1(f)], f~N(mu, v): conditional MEAN on the SI scale (plain back-transform gives a median and is biased low)."""
    m = np.maximum(mu, 0)
    return {'raw': m, 'sqrt': m**2 + v, 'q4': m**4 + 6*m**2*v + 3*v**2, 'log': np.expm1(np.minimum(m + v/2, 20))}[tf]

def krr(K, y, tri, evi, tf='log', alpha=.3, gp=False):
    f, finv = TF[tf]; yt = f(y[tri]); m = yt.mean(); a = yt.var() + 1e-12
    A = K[np.ix_(tri, tri)] + alpha * np.eye(len(tri)); Ke = K[np.ix_(evi, tri)]
    mu = m + Ke @ np.linalg.solve(A, yt - m)
    if not gp: return finv(mu)
    v = a * np.maximum(1 - np.einsum('ij,ji->i', Ke, np.linalg.solve(A, Ke.T)), 0) + a * alpha  # latent + noise variance
    return smear(tf, mu, v)

GP_KERNELS, GP_ALPHAS = ('mb2', 'mc2', 'mc3', 'mix'), (.1, .3)

def gp_members(K, y, tri, evi): return np.array([krr(K[k], y, tri, evi, 'log', a, gp=True) for k in GP_KERNELS for a in GP_ALPHAS])

def gp_log_ens_raw(K, y, tri, evi): return gp_members(K, y, tri, evi).mean(0)

def butina_groups(Kmat, thr):
    n = len(Kmat); d = []
    for i in range(1, n): d.extend(1 - Kmat[i, :i])
    cl = Butina.ClusterData(d, n, 1 - thr, isDistData=True, reordering=True)
    g = np.zeros(n, int)
    for j, c in enumerate(cl): g[list(c)] = j
    return g
