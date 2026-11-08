# VACV SI v23 — SMILES-only Kaggle baseline

**Task:** regress `SI VACV` from molecular structure. Competition MSE is computed on the **original SI scale**. `CC50_uM` and `IC50_uM_VACV` are explicitly forbidden: the code loads training **only** `SMILES` and `SI VACV`, test **only** `index` and `SMILES`; sample template **only** `index`. It never creates pseudo-labels from forbidden competition columns.

## Kaggle usage
1. Create a Kaggle notebook and import `VACV_SI_v23_Kaggle.ipynb`.
2. Add the original `train_v2.csv`, `test.csv`, `sample_submit.csv` as inputs.
3. CPU is enough. The notebook defaults to the embedded, reviewed ChEMBL snapshot, so Internet can be Off if the dependencies are installed. `SOURCE_MODE='download'` downloads public records from ChEMBL and needs Internet On.
4. Run All. Main file: `/kaggle/working/submission.csv`. Other options and diagnostics are in `/kaggle/working/vacv_v23/`.

## What was tested
- Official data: 320 train entries (312 unique canonical SMILES), 597 test entries (571 unique), 83 train SI=0. Maximum train SI=36066.667.
- KNN, hurdle (zero/nonzero + conditional SI), new feature kernels (feature Morgan, topological torsion, RDK pathway/physicochemical), plain KRR and GP. Best local family remains source-aware GP.
- v21 control: source-aware GP ensemble with mixed fingerprints.
- v23 GP: source-aware bit Morgan radius 2 Tanimoto, regularization alpha=0.1, with verified external SI.
- Both are evaluated via five-fold Murcko-scaffold and five-fold Butina 0.65 group splits, with source exclusions for held-out molecules and close analogues. MSE and MAE, plus diagnostic MSE excluding top SI, are reported.
- Additional cluster 0.50 and 0.75 validation is in `reports/v23_extra_cv.csv`; it compares new source GP against the old unsupervised-source GP reference, not full nested v21 blend.

| Scheme | v21 nested blend MSE | v23 all-sources MSE | v23 50% blend MSE |
|---|---:|---:|---:|
| Scaffold | 4,172,185 | 4,096,625 | 4,129,716 |
| Butina 0.65 | 4,086,330 | 3,970,040 | 4,026,429 |

**Caution:** the largest train SI accounts for most validation SSE. These results are from development splits used during model iteration; they are not independent confirmation and may not predict Kaggle Public/Private MSE. Aggressive GP produces some much larger test predictions than v21, so default `submission.csv` blends 50% v21 and 50% v23. Separate `submission_v23_aggressive.csv` and `submission_v21_control.csv` are saved.

## External data & integrity
- Audited ChEMBL pairs are sourced from independently published assays with provenance; no competition concentration columns enter the model.
- External reported SI and calculated SI are used only after source checking in `paired_data.py`.
- Public competition and external data terms may restrict redistribution. The public-ready project ZIP intentionally excludes attached competition train/test CSVs and model state holding training labels. The notebook rebuilds the state locally from your attached competition data.

## Saved-state prediction
After Run All, `vacv_v23/v23_model_state.json` contains the trained SI/SMILES states. To reproduce the 597 predictions without refitting:

```bash
python src/predict_v23.py --state /kaggle/working/vacv_v23/v23_model_state.json \
  --test test.csv --sample sample_submit.csv --output submission.csv
```

The saved-state reconstruction was verified locally (maximum absolute numerical difference 2.3e-13). This script only reads `index`, `SMILES` from the test file.
