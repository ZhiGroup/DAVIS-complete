
#%%
import pandas as pd
 
def fetch_finetuning_data(df: pd.DataFrame, split: str) -> tuple:

    # MSE results  
    test_mse_mean_wt_gt_across_protein = df['mean_test_mse_wt_groundtruth_baseline'].mean()
    test_mse_std_wt_gt_across_protein = df['mean_test_mse_wt_groundtruth_baseline'].std()

    test_mse_mean_wt_pred_across_protein = df['mean_test_mse_wt_prediction_baseline'].mean()
    test_mse_std_wt_pred_across_protein = df['mean_test_mse_wt_prediction_baseline'].std()

    test_mse_mean_pt_across_protein = df['mean_test_mse_original'].mean()
    test_mse_std_pt_across_protein = df['mean_test_mse_original'].std()

    test_mse_mean_finetuning_across_protein = df['mean_test_mse_finetuning'].mean()
    test_mse_std_finetuning_across_protein = df['mean_test_mse_finetuning'].std()


    # Pearson correlation results (rp)
    if split == 'same_mutation_different_drug':
        test_rp_mean_wt_gt_across_protein = df['mean_test_rp_wt_groundtruth_baseline'].mean()
        test_rp_std_wt_gt_across_protein = df['mean_test_rp_wt_groundtruth_baseline'].std()

        test_rp_mean_wt_pred_across_protein = df['mean_test_rp_wt_prediction_baseline'].mean()
        test_rp_std_wt_pred_across_protein = df['mean_test_rp_wt_prediction_baseline'].std()

    test_rp_mean_pt_across_protein = df['mean_test_rp_original'].mean()
    test_rp_std_pt_across_protein = df['mean_test_rp_original'].std()

    test_rp_mean_finetuning_across_protein = df['mean_test_rp_finetuning'].mean()
    test_rp_std_finetuning_across_protein = df['mean_test_rp_finetuning'].std()

    # C-index results
    if split == 'same_mutation_different_drug':
        test_cindex_mean_wt_gt_across_protein = df['mean_test_cindex_wt_groundtruth_baseline'].mean()
        test_cindex_std_wt_gt_across_protein = df['mean_test_cindex_wt_groundtruth_baseline'].std()

        test_cindex_mean_wt_pred_across_protein = df['mean_test_cindex_wt_prediction_baseline'].mean()
        test_cindex_std_wt_pred_across_protein = df['mean_test_cindex_wt_prediction_baseline'].std()

    test_cindex_mean_pt_across_protein = df['mean_test_cindex_original'].mean()
    test_cindex_std_pt_across_protein = df['mean_test_cindex_original'].std()
    
    test_cindex_mean_finetuning_across_protein = df['mean_test_cindex_finetuning'].mean()
    test_cindex_std_finetuning_across_protein = df['mean_test_cindex_finetuning'].std()
    

    if split == 'same_mutation_different_drug':
        return test_mse_mean_wt_gt_across_protein, test_mse_std_wt_gt_across_protein, \
               test_mse_mean_wt_pred_across_protein, test_mse_std_wt_pred_across_protein, \
               test_mse_mean_pt_across_protein, test_mse_std_pt_across_protein, \
               test_mse_mean_finetuning_across_protein, test_mse_std_finetuning_across_protein, \
               test_rp_mean_wt_gt_across_protein, test_rp_std_wt_gt_across_protein, \
               test_rp_mean_wt_pred_across_protein, test_rp_std_wt_pred_across_protein, \
               test_rp_mean_pt_across_protein, test_rp_std_pt_across_protein, \
               test_rp_mean_finetuning_across_protein, test_rp_std_finetuning_across_protein, \
               test_cindex_mean_wt_gt_across_protein, test_cindex_std_wt_gt_across_protein, \
               test_cindex_mean_wt_pred_across_protein, test_cindex_std_wt_pred_across_protein, \
               test_cindex_mean_pt_across_protein, test_cindex_std_pt_across_protein, \
               test_cindex_mean_finetuning_across_protein, test_cindex_std_finetuning_across_protein
    
    return test_mse_mean_wt_gt_across_protein, test_mse_std_wt_gt_across_protein, \
           test_mse_mean_wt_pred_across_protein, test_mse_std_wt_pred_across_protein, \
           test_mse_mean_pt_across_protein, test_mse_std_pt_across_protein, \
           test_mse_mean_finetuning_across_protein, test_mse_std_finetuning_across_protein, \
           test_rp_mean_pt_across_protein, test_rp_std_pt_across_protein, \
           test_rp_mean_finetuning_across_protein, test_rp_std_finetuning_across_protein, \
           test_cindex_mean_pt_across_protein, test_cindex_std_pt_across_protein, \
           test_cindex_mean_finetuning_across_protein, test_cindex_std_finetuning_across_protein


def fetch_pretraining_data(df: pd.DataFrame, split: str) -> tuple:
    
    # MSE results  
    all_mse_mean_wt_gt_across_protein = df['mean_all_mse_wt_groundtruth_baseline'].mean()
    all_mse_std_wt_gt_across_protein = df['mean_all_mse_wt_groundtruth_baseline'].std()

    all_mse_mean_wt_pred_across_protein = df['mean_all_mse_wt_prediction_baseline'].mean()
    all_mse_std_wt_pred_across_protein = df['mean_all_mse_wt_prediction_baseline'].std()

    all_mse_mean_pt_across_protein = df['mean_all_mse_original'].mean()
    all_mse_std_pt_across_protein = df['mean_all_mse_original'].std()

    # Pearson correlation results (rp)
    if split == 'same_mutation_different_drug':
        all_rp_mean_wt_gt_across_protein = df['mean_all_rp_wt_groundtruth_baseline'].mean()
        all_rp_std_wt_gt_across_protein = df['mean_all_rp_wt_groundtruth_baseline'].std()

        all_rp_mean_wt_pred_across_protein = df['mean_all_rp_wt_prediction_baseline'].mean()
        all_rp_std_wt_pred_across_protein = df['mean_all_rp_wt_prediction_baseline'].std()

    all_rp_mean_pt_across_protein = df['mean_all_rp_original'].mean()
    all_rp_std_pt_across_protein = df['mean_all_rp_original'].std()

    # C-index results
    if split == 'same_mutation_different_drug':
        all_cindex_mean_wt_gt_across_protein = df['mean_all_cindex_wt_groundtruth_baseline'].mean()
        all_cindex_std_wt_gt_across_protein = df['mean_all_cindex_wt_groundtruth_baseline'].std()

        all_cindex_mean_wt_pred_across_protein = df['mean_all_cindex_wt_prediction_baseline'].mean()
        all_cindex_std_wt_pred_across_protein = df['mean_all_cindex_wt_prediction_baseline'].std()

    all_cindex_mean_pt_across_protein = df['mean_all_cindex_original'].mean()
    all_cindex_std_pt_across_protein = df['mean_all_cindex_original'].std()


    if split == 'same_mutation_different_drug':
        return all_mse_mean_wt_gt_across_protein, all_mse_std_wt_gt_across_protein, \
               all_mse_mean_wt_pred_across_protein, all_mse_std_wt_pred_across_protein, \
               all_mse_mean_pt_across_protein, all_mse_std_pt_across_protein, \
               all_rp_mean_wt_gt_across_protein, all_rp_std_wt_gt_across_protein, \
               all_rp_mean_wt_pred_across_protein, all_rp_std_wt_pred_across_protein, \
               all_rp_mean_pt_across_protein, all_rp_std_pt_across_protein, \
               all_cindex_mean_wt_gt_across_protein, all_cindex_std_wt_gt_across_protein, \
               all_cindex_mean_wt_pred_across_protein, all_cindex_std_wt_pred_across_protein, \
               all_cindex_mean_pt_across_protein, all_cindex_std_pt_across_protein

    return all_mse_mean_wt_gt_across_protein, all_mse_std_wt_gt_across_protein, \
           all_mse_mean_wt_pred_across_protein, all_mse_std_wt_pred_across_protein, \
           all_mse_mean_pt_across_protein, all_mse_std_pt_across_protein, \
           all_rp_mean_pt_across_protein, all_rp_std_pt_across_protein, \
           all_cindex_mean_pt_across_protein, all_cindex_std_pt_across_protein


def fmt(mean, std, decimals=2):
    return f"{mean:.{decimals}f} ({std:.{decimals}f})"


round = 2

## Here is for different_mutation_same_drug

deepdta_df = pd.read_csv('docking_free_models/DeepDTA_result_finetuning_different_mutation_same_drug_epoch30_lr0.0005_combinationseedFalse.csv')
attentiondta_df = pd.read_csv('docking_free_models/AttentionDTA_result_finetuning_different_mutation_same_drug_epoch30_lr0.0005_combinationseedFalse.csv')
graphdta_df = pd.read_csv('docking_free_models/GraphDTA_result_finetuning_different_mutation_same_drug_epoch30_lr0.0005_combinationseedFalse.csv')
dgraphdta_df = pd.read_csv('docking_free_models/DGraphDTA_result_finetuning_different_mutation_same_drug_epoch30_lr0.0005_combinationseedFalse.csv')
mgraphdta_df = pd.read_csv('docking_free_models/MGraphDTA_result_finetuning_different_mutation_same_drug_epoch30_lr0.0005_combinationseedFalse.csv')
fda_df = pd.read_csv('affinity/GIGN/result_finetuning_different_mutation_same_drug_epoch30_lr0.005_combinationseedFalse.csv')
boltz2_df = pd.read_csv('/data/mwu11/boltz/result_finetuning_different_mutation_same_drug_epoch10_lr0.0003.csv')

models = {
    'DeepDTA': deepdta_df,
    'AttentionDTA': attentiondta_df,
    'GraphDTA': graphdta_df,
    'DGraphDTA': dgraphdta_df,
    'MGraphDTA': mgraphdta_df,
    'FDA': fda_df,
    'boltz2': boltz2_df
}


# Pretraining summary
pretrain_rows = []
for name, df_model in models.items():
    (
        p_mse_wt_gt_mean, p_mse_wt_gt_std,
        p_mse_wt_pred_mean, p_mse_wt_pred_std,
        p_mse_pt_mean, p_mse_pt_std,
        p_rp_pt_mean, p_rp_pt_std,
        p_cindex_pt_mean, p_cindex_pt_std
    ) = fetch_pretraining_data(df_model, split='different_mutation_same_drug')

    pretrain_rows.append({
        'Model': name,
        'MSE WT-GT': fmt(p_mse_wt_gt_mean, p_mse_wt_gt_std, round),
        'MSE WT-Pred': fmt(p_mse_wt_pred_mean, p_mse_wt_pred_std, round),
        'MSE Pretrain': fmt(p_mse_pt_mean, p_mse_pt_std, round),
        'RP Pretrain': fmt(p_rp_pt_mean, p_rp_pt_std, round),
        'CIndex Pretrain': fmt(p_cindex_pt_mean, p_cindex_pt_std, round)
    })

pretrain_summary = pd.DataFrame(pretrain_rows).set_index('Model')
print("\nPretraining Performance Summary for different_mutation_same_drug:")
print(pretrain_summary.to_string())


# Finetuning summary
rows = []
for model_name, df_model in models.items():
    (
        mse_wt_gt_mean, mse_wt_gt_std,
        mse_wt_pred_mean, mse_wt_pred_std,
        mse_pt_mean, mse_pt_std,
        mse_ft_mean, mse_ft_std,
        rp_pt_mean, rp_pt_std,
        rp_ft_mean, rp_ft_std,
        cindex_pt_mean, cindex_pt_std,
        cindex_ft_mean, cindex_ft_std
    ) = fetch_finetuning_data(df_model, split='different_mutation_same_drug')

    rows.append({
        'Model': model_name,
        'MSE WT-GT': fmt(mse_wt_gt_mean, mse_wt_gt_std, round),
        'MSE WT-Pred': fmt(mse_wt_pred_mean, mse_wt_pred_std, round),
        'MSE PT': fmt(mse_pt_mean, mse_pt_std, round),
        'MSE Finetune': fmt(mse_ft_mean, mse_ft_std, round),
        'RP PT': fmt(rp_pt_mean, rp_pt_std, round),
        'RP Finetune': fmt(rp_ft_mean, rp_ft_std, round),
        'CIndex PT': fmt(cindex_pt_mean, cindex_pt_std, round),
        'CIndex Finetune': fmt(cindex_ft_mean, cindex_ft_std, round)
    })

summary_df = pd.DataFrame(rows).set_index('Model')
print("Finetuning Performance Summary for different_mutation_same_drug:")
print(summary_df.to_string())

#%%
## Here is for same_mutation_different_drug

deepdta_df = pd.read_csv('docking_free_models/DeepDTA_result_finetuning_same_mutation_different_drug_epoch10_lr0.0001_combinationseedFalse.csv')
attentiondta_df = pd.read_csv('docking_free_models/AttentionDTA_result_finetuning_same_mutation_different_drug_epoch10_lr0.0001_combinationseedFalse.csv')
graphdta_df = pd.read_csv('docking_free_models/GraphDTA_result_finetuning_same_mutation_different_drug_epoch10_lr0.0001_combinationseedFalse.csv')
dgraphdta_df = pd.read_csv('docking_free_models/DGraphDTA_result_finetuning_same_mutation_different_drug_epoch10_lr0.0001_combinationseedFalse.csv')
mgraphdta_df = pd.read_csv('docking_free_models/MGraphDTA_result_finetuning_same_mutation_different_drug_epoch10_lr0.0001_combinationseedFalse.csv')
fda_df = pd.read_csv('affinity/GIGN/result_finetuning_same_mutation_different_drug_epoch10_lr0.0001_combinationseedFalse.csv')
boltz2_df = pd.read_csv('/data/mwu11/boltz/result_finetuning_same_mutation_different_drug_epoch10_lr0.0003.csv')


models = {
    'DeepDTA': deepdta_df,
    'AttentionDTA': attentiondta_df,
    'GraphDTA': graphdta_df,
    'DGraphDTA': dgraphdta_df,
    'MGraphDTA': mgraphdta_df,
    'FDA': fda_df,
    'boltz2': boltz2_df
    }

# Pretraining summary
pretrain_rows = []
for name, df_model in models.items():
    (
        p_mse_wt_gt_mean, p_mse_wt_gt_std,
        p_mse_wt_pred_mean, p_mse_wt_pred_std,
        p_mse_pt_mean, p_mse_pt_std,
        p_rp_wt_gt_mean, p_rp_wt_gt_std,
        p_rp_wt_pred_mean, p_rp_wt_pred_std,
        p_rp_pt_mean, p_rp_pt_std,
        p_cindex_wt_gt_mean, p_cindex_wt_gt_std,
        p_cindex_wt_pred_mean, p_cindex_wt_pred_std,
        p_cindex_pt_mean, p_cindex_pt_std
    ) = fetch_pretraining_data(df_model, split='same_mutation_different_drug')

    pretrain_rows.append({
        'Model': name,
        'MSE WT-GT': fmt(p_mse_wt_gt_mean, p_mse_wt_gt_std, round),
        'MSE WT-Pred': fmt(p_mse_wt_pred_mean, p_mse_wt_pred_std, round),
        'MSE Pretrain': fmt(p_mse_pt_mean, p_mse_pt_std, round),
        'RP WT-GT': fmt(p_rp_wt_gt_mean, p_rp_wt_gt_std, round),
        'RP WT-Pred': fmt(p_rp_wt_pred_mean, p_rp_wt_pred_std, round),
        'RP Pretrain': fmt(p_rp_pt_mean, p_rp_pt_std, round),
        'CIndex WT-GT': fmt(p_cindex_wt_gt_mean, p_cindex_wt_gt_std, round),
        'CIndex WT-Pred': fmt(p_cindex_wt_pred_mean, p_cindex_wt_pred_std, round),
        'CIndex Pretrain': fmt(p_cindex_pt_mean, p_cindex_pt_std, round)
    })

pretrain_summary = pd.DataFrame(pretrain_rows).set_index('Model')
print("\nPretraining Performance Summary for same_mutation_different_drug:")
print(pretrain_summary.to_string())

# Finetuning summary
rows = []
for model_name, df_model in models.items():
    (
        mse_wt_gt_mean, mse_wt_gt_std,
        mse_wt_pred_mean, mse_wt_pred_std,
        mse_pt_mean, mse_pt_std,
        mse_ft_mean, mse_ft_std,
        rp_wt_gt_mean, rp_wt_gt_std,
        rp_wt_pred_mean, rp_wt_pred_std,
        rp_pt_mean, rp_pt_std,
        rp_ft_mean, rp_ft_std,
        cindex_wt_gt_mean, cindex_wt_gt_std,
        cindex_wt_pred_mean, cindex_wt_pred_std,
        cindex_pt_mean, cindex_pt_std,
        cindex_ft_mean, cindex_ft_std
    ) = fetch_finetuning_data(df_model, split='same_mutation_different_drug')

    rows.append({
        'Model': model_name,
        'MSE WT-GT': fmt(mse_wt_gt_mean, mse_wt_gt_std, round),
        'MSE WT-Pred': fmt(mse_wt_pred_mean, mse_wt_pred_std, round),
        'MSE PT': fmt(mse_pt_mean, mse_pt_std, round),
        'MSE Finetune': fmt(mse_ft_mean, mse_ft_std, round),
        'RP WT-GT': fmt(rp_wt_gt_mean, rp_wt_gt_std, round),
        'RP WT-Pred': fmt(rp_wt_pred_mean, rp_wt_pred_std, round),
        'RP PT': fmt(rp_pt_mean, rp_pt_std, round),
        'RP Finetune': fmt(rp_ft_mean, rp_ft_std, round),
        'CIndex WT-GT': fmt(cindex_wt_gt_mean, cindex_wt_gt_std, round),
        'CIndex WT-Pred': fmt(cindex_wt_pred_mean, cindex_wt_pred_std, round),
        'CIndex PT': fmt(cindex_pt_mean, cindex_pt_std, round),
        'CIndex Finetune': fmt(cindex_ft_mean, cindex_ft_std, round)
    })

summary_df = pd.DataFrame(rows).set_index('Model')
print("Finetuning Performance Summary for same_mutation_different_drug:")
print(summary_df.to_string())


# %%
