import os
import pandas as pd


def map_expression_samples(expression_counts_df, cancer_column='Cancer_type'):
    expression_counts_df.index = expression_counts_df.index.astype(str)
    expression_counts_df.columns = expression_counts_df.columns.astype(str)

    # Step 1: Build short-form TCGA IDs from expression_df columns
    def get_short_tcga_id(full_id, project="LUAD"):
        parts = full_id.split('-')
        if len(parts) < 4:
            return None
        sample_type = parts[3][:2]
        sample_map = {
            '01': 'TP',
        '02': 'TR',
            '11': 'NT',
            '12': 'NB',
            '06': 'TM'
        }
        short_code = sample_map.get(sample_type, 'XX')
        return f"{project}-{parts[1]}-{parts[2]}-{short_code}"
    
    # Convert column headers
    short_sample_map = {
        row: get_short_tcga_id(row, cancer)
        for row, cancer in zip(expression_counts_df.index, expression_counts_df[cancer_column])
    }

    expression_counts_df.index = expression_counts_df.index.map(short_sample_map)
    return expression_counts_df


def read_expression_data(input_folder, cancer_types, cancer_column = 'Cancer_type', mean = False):
    mean_expression_counts_dict = {} if mean else []
    mean_expression_TPM_dict = {} if mean else []

    gene_name_mapping_df = pd.read_csv(os.path.join(input_folder, '../gencode.v36.annotation.gtf.gene.probemap'), sep='\t', index_col=0)

    for cancer_type in cancer_types:
        # Load data
        counts_df = pd.read_csv(os.path.join(input_folder, f'TCGA-{cancer_type}.star_counts.tsv.gz'), sep='\t', index_col=0).T
        tpm_df = pd.read_csv(os.path.join(input_folder, f'TCGA-{cancer_type}.star_tpm.tsv.gz'), sep='\t', index_col=0).T
        
        # Reverse log2 transform and convert to int
        counts_df = (2**counts_df - 1).astype(int)
        
        # Map gene names
        counts_df.columns = counts_df.columns + '|' + counts_df.columns.map(gene_name_mapping_df['gene'])
        tpm_df.columns = tpm_df.columns + '|' + tpm_df.columns.map(gene_name_mapping_df['gene'])
        
        counts_df.columns = counts_df.columns.str.split('|').str[1]
        tpm_df.columns = tpm_df.columns.str.split('|').str[1]
        counts_df[cancer_column] = cancer_type
        tpm_df[cancer_column] = cancer_type
        
        if mean:
            mean_expression_TPM_dict[cancer_type] = tpm_df.mean()
            mean_expression_counts_dict[cancer_type] = counts_df.mean()
        else:
            mean_expression_TPM_dict.append(tpm_df)
            mean_expression_counts_dict.append(counts_df)

    if mean:
        mean_expression_TPM_df = pd.DataFrame(mean_expression_TPM_dict)
        mean_expression_counts_df = pd.DataFrame(mean_expression_counts_dict)
    else:
        mean_expression_TPM_df = pd.concat(mean_expression_TPM_dict)
        mean_expression_counts_df = pd.concat(mean_expression_counts_dict)
        mean_expression_counts_df = map_expression_samples(mean_expression_counts_df, cancer_column=cancer_column)
        mean_expression_TPM_df = map_expression_samples(mean_expression_TPM_df, cancer_column=cancer_column)
    
    return (mean_expression_TPM_df, mean_expression_counts_df)