# DNA2RNA: Predicting Mutation Expression from DNA

## Introduction

Not all DNA mutations are expressed in RNA. Due to factors such as expression levels, some mutations present in the DNA may not be transcribed and therefore may not influence downstream processes like protein translation or immune recognition.

Understanding which mutations are actually expressed is crucial, especially in contexts like:
- Neoantigen presentation
- Tumor Mutational Burden (TMB)
- Immunotherapy response prediction

To address this, DNA2RNA is a machine learning-based tool that predicts whether a DNA mutation will appear in RNA. It uses paired DNA and RNA sequencing data to learn genomic and expression patterns that influence mutation expression.

## Goals

- Predict if a DNA mutation will appear in RNA using only DNA-based features.
- Predict allele frequency of expressed mutations.
- Use feature selection to identify important genomic factors.
- Build a pipeline for sample-level clinical predictions (e.g., TMB, survival, treatment response).

## Repository Overview

GitHub: https://github.com/asafpinhasitechnion/DNA2RNA_Scripts

This repository includes the Python scripts required to run the mutation expression prediction pipeline. It does not include the datasets.

### Main Components

| File                  | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `Run_Prediction.py`   | Entry point for running experiments. Parses command-line arguments, configures the run, and saves outputs. Useful for Condor or other job schedulers. |
| `Cross_Validation.py` | Implements cross-validation logic. Handles both regular and cross-cancer validation.          |
| `ML_models.py`        | Contains wrappers for scikit-learn, XGBoost, LightGBM, and PyTorch models. Standardizes interfaces. |
| `Feature_Selection.py`| Implements Recursive Feature Elimination (RFE) and manual feature selection.                  |
| `Utils.py`            | Utility functions for loading data, expression merging, and preprocessing.                    |
| `submit_condor.py`    | (Optional) Script template for launching multiple jobs on an HPC cluster using HTCondor.      |

## Usage

Run with:

```
python Run_Prediction.py \
  --model LightGBM \
  --cancer_type BRCA \
  --cv_mode normal \
  --n_estimators 200 \
  --learning_rate 0.05 \
  --max_depth 5 \
  --task binary
```

### Arguments

| Argument               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--model`              | One of LogisticRegression, RandomForest, DecisionTree, SVM, XGBoost, LightGBM, NeuralNet |
| `--cancer_type`        | Cancer type to subset from the dataset (e.g., BRCA, LUAD), or "all"         |
| `--cv_mode`            | "normal" (within-cancer) or "cross_cancer"                                  |
| `--n_estimators`       | Number of estimators (trees) for ensemble models                            |
| `--learning_rate`      | Learning rate for boosting models                                           |
| `--max_depth`          | Maximum depth of trees                                                      |
| `--task`               | Prediction task; currently only "binary" is supported                       |
| `--dont_use_expression`| If set, disables use of expression features                                 |

## Output

Results are saved in structured folders under `--output-folder`. These include:

- `results_<cancer>__/Mean/`: Mean accuracy and AUC
- `results_<cancer>__/Per_fold/`: Per-fold metrics
- `results_<cancer>__/Feature_importances/`: Feature importances per fold
- `rfe_auc_scores.csv`: AUC scores during feature elimination
- `rfe_selected_features.json`: Features retained at each stage

## Future Directions

- Expand the model to regression tasks (e.g., predicting allele frequency)
- Integrate with HLA-binding predictions and TMB panels
- Apply on clinical datasets to validate real-world applicability

## Requirements

- Python 3.8+
- scikit-learn
- xgboost
- lightgbm
- torch
- joblib
- pandas
- numpy
- matplotlib
- argparse

Install dependencies with:

```
pip install -r requirements.txt
```

## Example

To run RFE-based feature selection for LightGBM on all cancer types:

```
python Run_Prediction.py --model LightGBM --cancer_type all
```

To evaluate cross-cancer generalization:

```
python Run_Prediction.py --model XGBoost --cv_mode cross_cancer
```

## Author


