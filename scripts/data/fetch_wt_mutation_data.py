#%%
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import random


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

    mean_test_cindex = grep_value_after_word(file_path, 'mean test cindex:')
    std_test_cindex = grep_value_after_word(file_path, 'std test cindex:')
    mean_test_wt_cindex = grep_value_after_word(file_path, 'mean test_wt cindex:')
    std_test_wt_cindex = grep_value_after_word(file_path, 'std test_wt cindex:')
    mean_test_mutation_cindex = grep_value_after_word(file_path, 'mean test_mutation cindex:')
    std_test_mutation_cindex = grep_value_after_word(file_path, 'std test_mutation cindex:')

    return mean_test_mse, std_test_mse, mean_test_wt_mse, std_test_wt_mse, mean_test_mutation_mse, std_test_mutation_mse, mean_test_rp, std_test_rp, mean_test_wt_rp, std_test_wt_rp, mean_test_mutation_rp, std_test_mutation_rp, mean_test_cindex, std_test_cindex, mean_test_wt_cindex, std_test_wt_cindex, mean_test_mutation_cindex, std_test_mutation_cindex

def dict_result(_dict, model_name, ls_file_path):
    
    for split_method, file_path in zip(['wt_mutation'], ls_file_path):
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
        _dict[model_name][split_method]['mean_test_cindex'] = float(results[12])
        _dict[model_name][split_method]['std_test_cindex'] = float(results[13])
        _dict[model_name][split_method]['mean_test_wt_cindex'] = float(results[14])
        _dict[model_name][split_method]['std_test_wt_cindex'] = float(results[15])
        _dict[model_name][split_method]['mean_test_mutation_cindex'] = float(results[16])
        _dict[model_name][split_method]['std_test_mutation_cindex'] = float(results[17])

    return _dict

dict_model_splitmethod_results = defaultdict(lambda: defaultdict(dict))

file_path_deepdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/deepdta_davis_benchmark_{split_method}.log' for split_method in ['wt_mutation']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'DeepDTA', file_path_deepdta)

file_path_attentiondta = [f'/data/mwu11/DAVIS-complete/docking_free_models/attentiondta_davis_benchmark_{split_method}.log' for split_method in ['wt_mutation']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'AttentionDTA', file_path_attentiondta)

file_path_graphdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/graphdta_davis_benchmark_{split_method}.log' for split_method in ['wt_mutation']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'GraphDTA', file_path_graphdta)

file_path_dgraphdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/dgraphdta_davis_benchmark_{split_method}.log' for split_method in ['wt_mutation']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'DGraphDTA', file_path_dgraphdta)

file_path_mgraphdta = [f'/data/mwu11/DAVIS-complete/docking_free_models/mgraphdta_davis_benchmark_{split_method}.log' for split_method in ['wt_mutation']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'MGraphDTA', file_path_mgraphdta)

file_path_fda = [f'/data/mwu11/DAVIS-complete/affinity/GIGN/fda_davis_complete_{split_method}.log' for split_method in ['wt_mutation']]
dict_model_splitmethod_results = dict_result(dict_model_splitmethod_results, 'FDA', file_path_fda)



# %%
