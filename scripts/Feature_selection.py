            
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd        
from ML_models import SklearnModelWrapper  # or wherever it's defined


def merge_expression(mutation_df, expression_df, cancer_columns, gene_column, test_samples = None):
    if test_samples:
        temp_expression_df = expression_df.loc[:, ~expression_df.columns.isin(test_samples)]
    else:
        temp_expression_df = expression_df.copy()

    # Group by 'Cancer_type' and compute the mean for training genes
    temp_expression_df = temp_expression_df.groupby(cancer_columns).mean()

    # Reset index and melt into long format
    temp_expression_df = temp_expression_df.reset_index().melt(
        id_vars=cancer_columns, var_name=gene_column, value_name='Mean_Expression'
    )

    # Ensure both DataFrames have consistent data types for merging
    temp_expression_df[gene_column] = temp_expression_df[gene_column].astype(str)
    mutation_df[gene_column] = mutation_df[gene_column].astype(str)

    # Merge the two DataFrames on 'Cancer_type' and gene_column
    mutation_df = mutation_df.merge(
        temp_expression_df,
        on=[cancer_columns, gene_column]
    )
    return mutation_df




def RFE_feature_selection(model, mutation_df, expression_df, target_column, sample_column, gene_column, cancer_column, step=1, test_size=0.2, random_state=42):
    
    if isinstance(model, SklearnModelWrapper):
        model = copy.deepcopy(model.model)
    
    # chosen_features = [cancer_column, gene_column, target_column, sample_column, 'Tissue_lung', 'Transcript_Position','Start_position','COSMIC_total_alterations_in_gene','i_tumor_f', 'Tissue_skin', 'gc_content']
    mutation_df = merge_expression(mutation_df, expression_df, cancer_column, gene_column)
    # mutation_df = mutation_df[chosen_features]
    samples = mutation_df[sample_column].unique()
    train_samples, test_samples = train_test_split(
        samples, test_size=test_size, random_state=random_state
    )

    train_data = mutation_df[mutation_df[sample_column].isin(train_samples)].drop([cancer_column, gene_column, sample_column], axis=1)
    test_data = mutation_df[mutation_df[sample_column].isin(test_samples)].drop([cancer_column, gene_column, sample_column], axis=1)

    X_train, y_train = train_data.drop(target_column, axis=1), train_data[target_column]
    X_test, y_test = test_data.drop(target_column, axis=1), test_data[target_column]

    X = mutation_df.drop([cancer_column, gene_column, target_column, sample_column], axis=1)  
    y = mutation_df[target_column].values

    # Track scores
    scores = []
    features_dict = {}
    n_features_list = []

    # Start with all features
    remaining_features = list(X.columns)

    while len(remaining_features) >= step:
        print(len(remaining_features))
        # Train model
        model.fit(X_train[remaining_features], y_train)
        
        # Evaluate on test
        y_pred = model.predict_proba(X_test[remaining_features])[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        scores.append(auc)
        n_features_list.append(len(remaining_features))
        features_dict[len(remaining_features)] = copy.deepcopy(remaining_features)
        
        model.fit(X[remaining_features], y)
        # Get feature importances
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=remaining_features)
        least_important = feature_importance.nsmallest(step).index.tolist()
        for feat in least_important:
            remaining_features.remove(feat)

    # # Plot results
    # plt.figure(figsize=(10, 5))
    # plt.plot(n_features_list, scores, marker='o')
    # plt.gca().invert_xaxis()
    # plt.xlabel("Number of Features")
    # plt.ylabel("Accuracy on Test Set")
    # plt.title("Manual RFE: Accuracy vs. Number of Features")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return scores, n_features_list, features_dict