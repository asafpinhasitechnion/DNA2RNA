import json
import os
import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from Cross_Validation import cross_validate_with_expression, evaluate_train_test_split
from ML_models import PytorchModelWrapper, SklearnModelWrapper
from Utils import *
from Feature_selection import RFE_feature_selection

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DNA-to-RNA mutation prediction")

    parser.add_argument("--model", type=str, default="NeuralNet",
                        choices=["LogisticRegression", "RandomForest", "DecisionTree", "SVM", "XGBoost", "LightGBM", "NeuralNet"])
    parser.add_argument("--cancer_type", type=str, default="all")
    parser.add_argument("--input-folder", type=str, default=r"data")
    parser.add_argument("--output-folder", type=str, default=r"../results")
    parser.add_argument("--cv_mode", type=str, default="normal", choices=["normal", "cross_cancer"])
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass", "regression"])
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument('--layers', type=str, default='64,32', help="Comma-separated hidden layer sizes, e.g., '128,64,32'")

    parser.add_argument("--dont_use_expression", action="store_true")

    if len(os.sys.argv) > 1:
        return parser.parse_args()
    else:
        # Return default values if no arguments provided
        return parser.parse_args([])


def main(args):
    suffix_parts = []
    if args.learning_rate:
        suffix_parts.append(f"lr{args.learning_rate}")
    if args.n_estimators:
        suffix_parts.append(f"nest{args.n_estimators}")
    if args.max_depth:
        suffix_parts.append(f"md{args.max_depth}")
    if args.epochs:
        suffix_parts.append(f"epochs{args.epochs}")
    if args.cv_mode != 'normal':
        suffix_parts.append(f"{args.cv_mode}")
    model_name = args.model
    if model_name == 'NeuralNet':
        if not args.layers:
            raise ValueError("For NeuralNet, you must specify --layers with a comma-separated list of layer sizes.")
        model_name = model_name+'_'+args.layers.replace(',', '_')
    suffix_parts.append(f"{model_name}")
    suffix_parts.append(f"{args.task}")

    suffix = "__" + "__".join(suffix_parts) if suffix_parts else ""
    task = args.task
    if task != 'binary':
        raise ValueError(f"Only 'binary' is currently supported.")
    # Define base models
    models = {
        "LogisticRegression": SklearnModelWrapper(LogisticRegression(max_iter=1000), task=task),
        "RandomForest": SklearnModelWrapper(RandomForestClassifier(
            n_estimators=args.n_estimators or 100,
            max_depth=args.max_depth,
            random_state=42), task=task),
        "DecisionTree": SklearnModelWrapper(DecisionTreeClassifier(
            max_depth=args.max_depth,
            random_state=42), task=task),
        "SVM": SklearnModelWrapper(LinearSVC(max_iter=10000), task=task),
        "XGBoost": SklearnModelWrapper(XGBClassifier(
            n_estimators=args.n_estimators or 100,
            learning_rate=args.learning_rate or 0.1,
            max_depth=args.max_depth or 6,
            random_state=42), task=task),
        "LightGBM": SklearnModelWrapper(LGBMClassifier(
            n_estimators=args.n_estimators or 100,
            learning_rate=args.learning_rate or 0.1,
            max_depth=args.max_depth,
            random_state=42, verbosity = args.verbosity), task=task)
    }

    
    input_folder = args.input_folder
    data_folder = r"C:\Users\KerenYlab.MEDICINE\OneDrive - Technion\Asaf\Data\RNA\TCGA\Xena\tcga_xena_data"
    mutation_data_path = os.path.join(input_folder, 'TCGA_mutations.csv')
    print(f"Loading mutation data from '{mutation_data_path}'...")
    df = pd.read_csv(mutation_data_path, index_col=0)
    cancer_types = list(df['Cancer_type'].unique())

    cache_path = os.path.join(input_folder, 'cached_expression_df.joblib')
    if os.path.exists(cache_path):
        print(f"Loading cached expression data from '{cache_path}'...")
        expression_df = joblib.load(cache_path)
    else:
        print(f'Loading expression data from {data_folder}...')
        _, expression_df = read_expression_data(
            input_folder=data_folder,
            cancer_types=cancer_types,
            cancer_column='Cancer_type',
            mean=False
        )
        print(f"Caching expression data to '{cache_path}'...")
        joblib.dump(expression_df, cache_path)


    if args.cancer_type.lower() == "all":
        filtered_df = df
        cancer_type = "All"
    else:
        filtered_df = df[df['Cancer_type'] == args.cancer_type]
        if filtered_df.empty:
            raise ValueError(f"No data available for cancer type {args.cancer_type}.")
        cancer_type = args.cancer_type

    input_dim = filtered_df.shape[1] - 3

    nn_architecture = None
    if args.model == 'NeuralNet' and args.layers:
        nn_architecture = list(map(int, args.layers.split(',')))
    
    models["NeuralNet"] = PytorchModelWrapper(
        input_dim=input_dim,
        layers=nn_architecture,
        task='binary',
        learning_rate=args.learning_rate or 0.005,
        epochs=args.epochs or 25
    )

    if args.model not in models:
        raise ValueError(f"Model {args.model} not defined.")

    model = models[args.model]

    results_folder = os.path.join(args.output_folder, f'results_{cancer_type}{suffix}')
    os.makedirs(results_folder, exist_ok=True)
    fi_folder = os.path.join(results_folder, 'Feature_importances')
    fold_folder = os.path.join(results_folder, 'Per_fold')
    mean_results = os.path.join(results_folder, 'Mean')
    for folder in [fi_folder, fold_folder, mean_results]:
        os.makedirs(folder, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        fs_results = RFE_feature_selection(
            model=model,
            mutation_df=filtered_df.copy(),
            expression_df=expression_df.copy(),
            target_column='Appears_in_rna',
            sample_column='Tumor_Sample_Barcode',
            gene_column='Hugo_Symbol',
            cancer_column='Cancer_type',
            step=1,
            test_size=0.2,
            random_state=42,
        )

        scores, n_features_list, features_dict = fs_results
        # Save n_features_list and scores as CSV
        df_scores = pd.DataFrame({
            "n_features": n_features_list,
            "roc_auc": scores
        })

        df_scores.to_csv(os.path.join(results_folder, 'rfe_auc_scores.csv'), index=False)

        # Save features_dict as JSON
        with open(os.path.join(results_folder, 'rfe_selected_features.json'), "w") as f:
            json.dump(features_dict, f, indent=4)

    print(f"Running cross-validation...")
    results = cross_validate_with_expression(
        dataframe=filtered_df,
        expression_df=expression_df,
        target_column='Appears_in_rna',
        sample_column='Tumor_Sample_Barcode',
        gene_column='Hugo_Symbol',
        model=model,
        cv_mode=args.cv_mode,
        use_expression=not args.dont_use_expression,
        n_splits=5,
        random_state=42,
        verbose=args.verbosity
    )


    metrics = {
        "Model": args.model,
        "Cancer_Type": cancer_type,
        "Mean_Accuracy": results["mean_accuracy"],
        "Mean_ROC_AUC": results["mean_roc_auc"]
    }

    results_file = os.path.join(mean_results, f"results_{cancer_type}{suffix}.csv")
    pd.DataFrame([metrics]).to_csv(results_file)

    fold_df = results["fold_results"]
    fold_results_file = os.path.join(fold_folder, f"fold_results_{cancer_type}{suffix}.csv")
    fold_results_df = pd.DataFrame(fold_df)
    fold_results_df.index = 'Fold ' + fold_results_df.index.astype(str)
    fold_results_df.loc['Mean'] = fold_results_df.mean()
    fold_results_df.to_csv(fold_results_file)
    print(f"Per-fold results saved to '{fold_results_file}'.")

    feature_importances = results.get("feature_importances", [None])
    if feature_importances and any(fi is not None for fi in feature_importances):
        fi_file = os.path.join(fi_folder, f"feature_importances_{cancer_type}{suffix}.csv")
        fi_df = pd.DataFrame(feature_importances).T
        fi_df.index = 'Fold ' + fi_df.index.astype(str)
        fi_df.loc['Mean_Feature_Importance'] = fi_df.mean()
        fi_df = fi_df.sort_values('Mean_Feature_Importance', axis=1, ascending=False)
        fi_df.to_csv(fi_file)
        print(f"Feature importances saved to '{fi_file}'.")

    print(f"Job completed. Results saved to '{results_file}'.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
