
#%%
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import random
# import seaborn as sns


def grep_value_after_word(file_path, word):
    with open(file_path, 'r') as file:
        content = file.read()
        pattern = r'{} (\S+)'.format(word)  # Match the word followed by a space and capture the value
        match = re.search(pattern, content)
        if match:
            return match.group(1)  # Return the captured value
        else:
            return np.nan  # Return np.nan if no match is found

def grep_results(file_path):
    mean_test_mse = grep_value_after_word(file_path, 'mean test mse:')
    std_test_mse = grep_value_after_word(file_path, 'std test mse:')
    mean_test_wt_mse = grep_value_after_word(file_path, 'mean test_wt mse:')
    std_test_wt_mse = grep_value_after_word(file_path, 'std test_wt mse:')
    mean_test_mutation_mse = grep_value_after_word(file_path, 'mean test_mutation mse:')
    std_test_mutation_mse = grep_value_after_word(file_path, 'std test_mutation mse:')
    
    mean_test_rp = grep_value_after_word(file_path, 'mean test rp:')
    std_test_rp = grep_value_after_word(file_path, 'std test rp:')
    mean_test_wt_rp = grep_value_after_word(file_path, 'mean test_wt rp:')
    std_test_wt_rp = grep_value_after_word(file_path, 'std test_wt rp:')
    mean_test_mutation_rp = grep_value_after_word(file_path, 'mean test_mutation rp:')
    std_test_mutation_rp = grep_value_after_word(file_path, 'std test_mutation rp:')

    return mean_test_mse, std_test_mse, mean_test_wt_mse, std_test_wt_mse, mean_test_mutation_mse, std_test_mutation_mse, mean_test_rp, std_test_rp, mean_test_wt_rp, std_test_wt_rp, mean_test_mutation_rp, std_test_mutation_rp

def dict_result(_dict, model_name, ls_file_path):
    
    for split_method, file_path in zip(['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid'], ls_file_path):
        results = grep_results(file_path)
        _dict[model_name][split_method]['mean_test_mse'] = float(results[0])
        _dict[model_name][split_method]['std_test_mse'] = float(results[1])
        _dict[model_name][split_method]['mean_test_wt_mse'] = float(results[2])
        _dict[model_name][split_method]['std_test_wt_mse'] = float(results[3])
        _dict[model_name][split_method]['mean_test_mutation_mse'] = float(results[4])
        _dict[model_name][split_method]['std_test_mutation_mse'] = float(results[5])
        _dict[model_name][split_method]['mean_test_rp'] = float(results[6])
        _dict[model_name][split_method]['std_test_rp'] = float(results[7])
        _dict[model_name][split_method]['mean_test_wt_rp'] = float(results[8])
        _dict[model_name][split_method]['std_test_wt_rp'] = float(results[9])
        _dict[model_name][split_method]['mean_test_mutation_rp'] = float(results[10])
        _dict[model_name][split_method]['std_test_mutation_rp'] = float(results[11])

    return _dict

dict_model_splitmethod_results = defaultdict(lambda: defaultdict(dict))




file_path_deepdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/deepdta_davis_benchmark_{split_method}.log' for split_method in ['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'DeepDTA', file_path_deepdta)

file_path_attentiondta = [f'/data/mwu11/DAVIS-complete/docking_free_models/attentiondta_davis_benchmark_{split_method}.log' for split_method in ['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'AttentionDTA', file_path_attentiondta)

file_path_graphdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/graphdta_davis_benchmark_{split_method}.log' for split_method in ['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'GraphDTA', file_path_graphdta)

file_path_dgraphdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/dgraphdta_davis_benchmark_{split_method}.log' for split_method in ['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'DGraphDTA', file_path_dgraphdta)

file_path_mgraphdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/mgraphdta_davis_benchmark_{split_method}.log' for split_method in ['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'MGraphDTA', file_path_mgraphdta)

file_path_fda = [f'/data/mwu11/DAVIS-complete/affinity/GIGN/fda_davis_complete_{split_method}.log' for split_method in ['drug_name', 'drug_structure', 'protein_modification_drug_name', 'protein_modification', 'protein_name', 'protein_seqid_drug_structure', 'protein_seqid']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'FDA', file_path_fda)

# %%
