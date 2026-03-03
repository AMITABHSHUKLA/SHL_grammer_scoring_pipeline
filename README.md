# **Grammar Scoring Engine for Spoken Responses**

## Overview

This repository presents a production-grade machine learning pipeline for automated grammar scoring of spoken audio responses.

The solution integrates deep acoustic modeling, classical signal processing, semantic language embeddings, and ensemble learning to produce a robust regression model for grammar assessment.

The final system achieved:

- **Leaderboard Rank:** 1st Place  
- **Leaderboard RMSE:** 0.450  
- **Improved Score with CoLA Embeddings:** 0.447  

This project demonstrates the effectiveness of multi-view feature representation combined with systematic model optimization and stacking.

---

## Problem Statement

Given spoken audio responses, the objective is to predict a continuous grammar score. The challenge involves:

- Extracting meaningful signals from raw speech
- Modeling both acoustic delivery and linguistic correctness
- Preventing overfitting in a high-dimensional, small-sample setting
- Achieving stable generalization under cross-validation

---

## Technical Architecture

### 1. Hybrid Feature Representation

Each audio sample is transformed into a high-dimensional feature vector composed of:

#### Deep Acoustic Embeddings
- Extracted from Whisper encoder
- Log-Mel spectrogram → Transformer encoder → Mean pooling
- Captures prosody, fluency, articulation, and temporal dynamics

#### Handcrafted Signal Features
- MFCC statistics
- Zero Crossing Rate
- RMS Energy
- Provides low-dimensional, stable signal characteristics

#### Semantic Text Embeddings
- Whisper transcription
- Sentence Transformer (MPNet)
- Captures grammar structure, coherence, and vocabulary usage

#### Extended Linguistic Features
- CoLA embeddings for grammatical acceptability
- Improved leaderboard performance from 0.450 to 0.447

#### Audio Duration
- Provides temporal context

All components are concatenated into a unified hybrid embedding.

---

## Model Development Strategy

### Cross-Validation
- 5-fold KFold with shuffle
- RMSE as the primary evaluation metric
- Strict leakage prevention through fold-level scaling

### Hyperparameter Optimization
- Bayesian optimization using Optuna
- Tuned models:
  - CatBoost (GPU)
  - XGBoost (GPU)
  - SVR (RBF Kernel)

### Ablation Study
Evaluated:
- Tree-based models
- Kernel methods (SVR)
- Linear regularized models

This ensured data-driven model selection rather than arbitrary stacking.

---

## Ensemble Architecture

Final architecture uses stacking regression:

**Base Models**
- CatBoost
- Scaled SVR

**Meta-Learner**
- Ridge Regression (RidgeCV)

Stacking combines:
- Non-linear interaction modeling from trees
- Smooth geometric boundaries from SVR

This reduced both bias and variance and produced the best cross-validated performance.

---

## Class Balancing Strategy

To address score imbalance:

- Applied noise-based oversampling for minority score classes
- Added controlled Gaussian perturbation
- Improved distribution uniformity while preserving feature structure
- Re-evaluated ensembles under balanced training conditions

---

## Final Pipeline

1. Load metadata and construct audio paths  
2. Extract hybrid feature representations  
3. Perform hyperparameter optimization  
4. Conduct ablation and model comparison  
5. Build stacking ensembles  
6. Evaluate via 5-fold CV (RMSE reported)  
7. Train final model on full dataset  
8. Generate predictions and clip to valid range  

---

## Results

| Model Variant | Leaderboard RMSE |
|--------------|------------------|
| Hybrid + Stacking | 0.450 |
| Hybrid + CoLA + Stacking | 0.447 |

Final ranking: **1st Place**

---

## Key Insights

- Grammar scoring benefits from combining acoustic and linguistic modalities.
- Kernel methods remain highly competitive in high-dimensional embedding spaces.
- Feature diversity contributes more than increasing model complexity alone.
- Proper cross-validation and leakage prevention are critical in small datasets.
- Stacking complementary model families improves stability.

---

## Technology Stack

- PyTorch
- Whisper
- Sentence Transformers
- Librosa
- Scikit-learn
- CatBoost
- XGBoost
- Optuna
- NumPy / Pandas

---

## Reproducibility

- Fixed random seeds
- Deterministic cross-validation
- Explicit hyperparameter definitions
- Clear pipeline structure

The notebook can be executed end-to-end to reproduce training RMSE and final predictions.

---
