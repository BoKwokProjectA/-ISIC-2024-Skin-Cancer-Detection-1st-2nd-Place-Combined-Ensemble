# ISIC 2024 Skin Cancer Detection — 1st & 2nd Place Solutions

**Competition**: [ISIC 2024 - Skin Cancer Detection with 3D Total Body Photography](https://www.kaggle.com/competitions/isic-2024-challenge)  
**Metric**: Partial AUC (pAUC) at 80% TPR (official competition metric)  
**Team**: Solo (Bo Kwok)

## Final Results (Partial AUC)

| Solution                  | Partial AUC (pAUC) | Rank      | Notes |
|---------------------------|--------------------|-----------|----------------------------------------------------------|
| **1st Place (Original)**  | **0.17706**        | 1st       | Pure tabular CatBoost with extensive feature engineering |
| **2nd Place (Original)**  | Top-2 (private)    | 2nd       | Multi-model CV + GBDT meta-learner |
| **Combined Ensemble**     | **0.19553**        | —         | Fusion of both solutions (this notebook) |

The combined version improves the 1st-place score by **+0.01847** pAUC through the integration of strong DNN predictions from 
the 2nd-place solution.

---

## Solution Overview

This repository contains:
- Adapted original 1st-place and 2nd-place notebooks (originally developed in the Kaggle environment) to run efficiently in Google Colab
- A new **combined ensemble** that merges the strengths of both approaches

The solution is a **hybrid CV + tabular ensemble**:
- Multiple high-performing image models (from 2nd place) generate powerful meta-features
- These are fed into an extremely rich tabular pipeline (from 1st place) with patient-level normalization, clustering, outlier detection,
  and OOF fusion

---

## 1st Place Solution (Tabular-Focused)

**Core idea**: Hand-crafted domain-specific features + patient-level statistics dominate medical tabular modeling.

**Key techniques used**:
- 34 original numeric columns + **50+ engineered features** (`new_num_cols`)
- **Patient-normalized features** (`_patient_norm`) for every numeric and engineered column
- Special patient-level aggregates (`count_per_patient`, `tbp_lv_areaMM2_patient`, `tbp_lv_areaMM2_bp`)
- **Local Outlier Factor (LOF)** per patient on selected features
- KMeans clustering (29 clusters) on scaled features → cluster-relative features (`__cluster`)
- OOF predictions from two strong image models (`oof_eva_score`, `oof_edgenext_score`) as meta-features
- StratifiedGroupKFold (patient-aware)
- RandomOverSampler + RandomUnderSampler pipeline
- Optuna-tuned **CatBoost** (GPU) with custom pAUC metric

**Final CV pAUC**: **0.17706**

---

## 2nd Place Solution (CV + Meta-Learner)

**Core idea**: Diverse computer-vision backbones + tabular GBDT on top of DNN predictions.

**Key techniques used**:
- Multiple image models trained with Lightning:
  - BEiT v2 base
  - SwinV2 small
  - EVA-02 small
  - DeiT3 small
  - ResNeXt50
  - (plus several tip-finetune variants)
- Albumentations augmentations (including GaussNoise, CoarseDropout, etc.)
- Patient-aware cross-validation
- DNN predictions saved and later used as high-signal tabular features
- Final GBDT (CatBoost/LightGBM) meta-learner on tabular features + DNN OOF

---

## Combined Ensemble (This Repository)

The combined notebook executes in the following order. I clearly mark which parts come from the **1st-place** solution and 
which come from the **2nd-place** solution.

### Code Execution Flow & Component Origin

1. **Setup & Dependencies**  
   → Mostly from 2nd place (Lightning, Hydra, rootutils, torch, etc.)

2. **DNN Inference Pipeline**
   → **Entirely from 2nd place**  
   - Loads multiple pre-trained image models  
   - Runs full inference on test set using Lightning DataModule + custom transforms  
   - Saves predictions for: `0821-beitv2_base`, `0824-swinv2_small`, `0824-eva02_small`, `0827-deit3_small`, `0828-resnext50`, etc.  
   - These DNN predictions become the most important meta-features in the final model.

3. **Tabular Data Loading & Feature Engineering**
   → **Core from 1st place** (the most important part)  
   - `read_data()` + all 50+ `new_num_cols`  
   - Patient normalization (`_patient_norm`) for every numeric/engineered column  
   - `count_per_patient`, area sums per patient and per body part  
   - **LOF outlier score** per patient  
   - OOF merging from EVA and EdgeNext image models (`oof_eva_score`, `oof_edgenext_score`)  
   - KMeans clustering (29 clusters) + cluster-relative features (`__cluster`)  
   - One-hot encoding of categorical columns

4. **Merging DNN Predictions**
   → **From 2nd place**  
   - All DNN OOF columns from step 2 are merged into the 1st-place feature set  
   - Missing DNN columns get placeholder value 0.0

5. **Final CatBoost Training & Prediction**
   → **From 1st place** (model architecture & training loop)  
   - Same CatBoost hyperparameters as 1st place  
   - Same sampling strategy (RandomOverSampler + RandomUnderSampler)  
   - Same StratifiedGroupKFold + custom pAUC metric  
   - Final inference on test set

**Result**: Combined validation pAUC = **0.19553** (significant lift over 1st place’s 0.17706)

---

## All Major Components & Libraries Used

- **Data**: Polars, Pandas
- **Feature Engineering**: NumPy, Scikit-learn (StandardScaler, KMeans, LocalOutlierFactor), custom patient-norm & cluster features
- **Modeling**:
  - CatBoost (GPU)
  - PyTorch + Lightning (for DNN inference)
- **Cross-validation**: StratifiedGroupKFold
- **Sampling**: imbalanced-learn (RandomOverSampler + RandomUnderSampler)
- **Optimization**: Optuna (used in 1st place)
- **Metric**: Custom partial AUC implementation
- **Other**: Hydra (config), rootutils, joblib

---

## Repository Contents

- `notebooks/1st_place_solution.ipynb` — Adapted 1st-place solution (originally for Kaggle, now runs in Google Colab)
- `notebooks/2nd_place_solution.ipynb` — Adapted 2nd-place solution (originally for Kaggle, now runs in Google Colab)
- `notebooks/combined_ensemble.ipynb` — New combined ensemble that integrates the strengths of both solutions (recommended)


