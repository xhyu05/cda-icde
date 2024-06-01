#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import torch
import csv
import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from collections import Counter
#from utils.data_loaders import dataset_loader

from utils.utils import *
from utils.mab import UCBMultiArmBandit

## Setting up and initialization

# Fix the seed ------------------------------------------------------
np.random.seed(42)
multi_imp_num = 1
multi_imp_iter = 1


# Function produce_NA for generating missing values 
# Using functions in utils & mf.ampute_data
# utils is downloaded by
#    import wget
#    wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')
#    reference: https://github.com/AnotherSamWilson/miceforest#Effects-of-Mean-Matching
#               https://github.com/BorisMuzellec/MissingDataOT


def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    "real_missing_rate" : the real rate of missing values
    """
    
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    #return {'mask': mask, 'real_missing_rate': (mask.sum()).numpy()/np.prod(mask.size())}
    return mask, X_nas.numpy(), (mask.sum()).numpy()/np.prod(mask.size())

# transform a table (dataframe) into numpy matrix
def df_2_array(df_original):

    df = df_original.copy()
    ks = df.keys()
    column_mapping_relations = []
    for k in ks:
        datatype = df[k].dtype
        if  datatype != 'int64' and datatype != 'float64':
            b = pd.get_dummies(df[k])
            categories = b.keys()
            num_2_category = dict(zip(range(len(categories)), categories))
            df[k] = b.values.argmax(1)
            df[k] = df[k].astype('category')
        else:
            num_2_category = {'max':max(df[k]), 'min':min(df[k])}

        column_mapping_relations.append({'datatype':df[k].dtype, 'mapping':num_2_category, 'col':k})

    return df, column_mapping_relations


def get_avg_error(df_ground, df_imputed, mapping):
    diff = (df_ground.astype('float64') - df_imputed.astype('float64')).abs()
    s = 0.0
    for info in mapping:
        if info['datatype']=='category':
            s += (diff[info['col']]>1e-6).astype(int).sum()
        else:
            s += (diff[info['col']].sum()/(info['mapping']['max']-info['mapping']['min']))
            
    return s / diff.shape[0] / diff.shape[1]


# get incomplete table using mask matrix
# mask is a torch tensor
# df_complete is a dataframe
def get_incomplete_table(df_complete, mask):
    df_miss = df_complete.copy()
    df_mask = pd.DataFrame(1-mask).astype('int64').replace(0, np.nan)
    missing_indexes = np.where(np.asanyarray(np.isnan(df_mask)))

    length = len(missing_indexes[0])
    for i in range(length):
        df_miss.iat[missing_indexes[0][i], missing_indexes[1][i]] = pd.NA

    return df_miss, missing_indexes


def random_value_acquisition(df_miss, df_arr, budget, queries, W):
    miss_rows, miss_cols = np.where(np.isnan(df_miss.to_numpy()))
    budget_unit = int(len(miss_rows)*budget)
    acq_indexes = random.sample(range(0, len(miss_rows)), budget_unit)
    acquired_df_miss = df_miss.copy()

    for i in acq_indexes:
        acquired_df_miss.iat[miss_rows[i], miss_cols[i]] = df_arr.iat[miss_rows[i], miss_cols[i]]
        
    return acquired_df_miss


def random_sample_acquisition(df_miss, df_arr, budget, query):
    df_miss_copy = df_miss.copy()
    rids = random.sample(range(len(df_miss)), int(len(df_miss)*budget))
    df_miss_copy.iloc[rids] = df_arr.iloc[rids]
    
    return df_miss_copy

def read_menu():
    # Specify the file path
    file_path = 'menu.csv'
    # Read the data from the CSV file
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        menu_column = [list(map(str, row)) for row in csv_reader]
    
    return menu_column

def greedy_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    columns = []
    for query in queries:
        query_columns = [q['col'] for q in query]
        columns += query_columns

    total_acq_num = budget*df_miss.isna().sum().sum()      # get acquisition cell number
    columns_count = dict(Counter(columns))
    for k, v in columns_count.items():
        _rows = np.where(np.isnan(df_miss_copy[k].to_numpy())) # get rows for missing column k
        sample_num = int(v / len(columns)*total_acq_num)       # get acquisition cell number in column k
        sample_rows = random.sample(range(len(_rows[0])), sample_num)
        for r in sample_rows:
            df_miss_copy[k].iat[_rows[0][r]] = df_arr[k].iat[_rows[0][r]]
    
    return df_miss_copy


def mab_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    columns = []
    for query in queries:
        query_columns = [q['col'] for q in query]
        columns += query_columns
    columns_count = dict(Counter(columns))
    initial_rewards = list(columns_count.values())
    arm_labels = list(columns_count.keys())

    # initial UCB
    num_arms = len(arm_labels)
    bandit = UCBMultiArmBandit(num_arms, initial_rewards)

    # Simulate bandit plays
    batch_size_per_iter = 20 # change it according to your need
    num_plays = int(budget*df_miss.isna().sum().sum() / batch_size_per_iter)

    for _ in range(num_plays):
        chosen_arm = bandit.select_arm()

        # the reward should be if it has found a usable tuple
        reward = 0
        col = arm_labels[chosen_arm]
        missing_index = np.where(df_miss_copy[col].isna())[0]
        acquisition_index = np.random.choice(missing_index, size=batch_size_per_iter, replace=False)
        for ind in acquisition_index:
            df_miss_copy[col].iat[ind] = df_arr[col].iat[ind]
        imputed_part = df_miss_copy.loc[acquisition_index]
        for i, query in enumerate(queries):
            # get the dataframe of acquired answer and ground truth
            data_product = query_on_df(query, imputed_part)
            reward += len(data_product)*W[i]

        bandit.update(chosen_arm, reward)
    
    return df_miss_copy


def one_pass_max_utility(df_miss, df_arr, budget, query):
    df_miss_copy = df_miss.copy()
    _rows, _cols = np.where(np.isnan(df_miss.to_numpy()))
    budget_units = int(budget*len(_rows))+1
    query_columns = [q['col_num'] for q in query]

    candidate_rows = get_candidate_rows(query_columns, _rows, _cols)

    # Create kernels. 
    kernel = mf.ImputationKernel(
      data=df_miss_copy,
      datasets=5,
      save_all_iterations=False,
      random_state=1991,
    )

    # Run the MICE algorithm for 3 iterations on each of the datasets
    kernel.mice(3, verbose=True, min_data_in_leaf = 6)
    acq_task = get_utility_score(kernel, query, candidate_rows, budget_units)

    for line in acq_task:
        for c in query_columns:
            if pd.isna(df_miss_copy.iat[line[0], c])==True:
                budget_units -= 1
                df_miss_copy.iat[line[0],c] = df_arr.iat[line[0],c]

    if budget_units>0:
        _rows, _cols = np.where(np.isnan(df_miss_copy.to_numpy()))
        candidate_rows = get_candidate_rows(query_columns, _rows, _cols)
        acq_task = random.sample(candidate_rows, budget_units)
        i = 0
        while budget_units>0:
            for c in query_columns:
                if pd.isna(df_miss_copy.iat[acq_task[i], c])==True:
                    df_miss_copy.iat[acq_task[i], c] = df_arr.iat[acq_task[i], c]
                    budget_units -= 1
            i += 1
            if budget_units-query_on_df(query, df_miss_copy).isna().sum().sum()<=0:
                for ind in query_on_df(query, df_miss_copy).index.values.tolist():
                    for c in range(df_miss_copy.shape[1]):
                         if pd.isna(df_miss_copy.iat[ind,c])==True:
                                df_miss_copy.iat[ind, c] = df_arr.iat[ind, c] 
                budget_units = 0
    
    return df_miss_copy


def one_pass_max_uncertainty(df_miss, df_arr, budget, query):
    df_miss_copy = df_miss.copy()
    _rows, _cols = np.where(np.isnan(df_miss.to_numpy()))
    budget_units = int(budget*len(_rows))+1
    query_columns = [q['col_num'] for q in query]

    candidate_rows = get_candidate_rows(query_columns, _rows, _cols)

    # Create kernels. 
    kernel = mf.ImputationKernel(
      data=df_miss_copy,
      datasets=5,
      save_all_iterations=False,
      random_state=1991,
    )

    # Run the MICE algorithm for 3 iterations on each of the datasets
    kernel.mice(3, verbose=True, min_data_in_leaf = 6)
    acq_task = get_uncertainty_score(kernel, query, candidate_rows)
    
    i = 0
    while budget_units>0:
        for c in query_columns:
            if pd.isna(df_miss_copy.iat[acq_task[i], c])==True:
                df_miss_copy.iat[acq_task[i], c] = df_arr.iat[acq_task[i], c]
                budget_units -= 1
        i += 1
        if budget_units-query_on_df(query, df_miss_copy).isna().sum().sum()<=0:
                rids = query_on_df(query, df_miss_copy).index.values.tolist()
                df_miss_copy.iloc[rids] = df_arr.iloc[rids]
                budget_units = 0
    
    return df_miss_copy

    
# # get incomplete table in numerical form
# def get_incomplete_matrix(df_complete, mask):
#     x_miss = np.copy(df_complete.to_numpy())
#     x_miss = x_miss.astype('float32')
#     mask = mask.numpy() >0.5
#     x_miss[mask] = np.nan

#     df_mask = pd.DataFrame(1-mask).astype('int64').replace(0, np.nan)
#     missing_indexes = np.where(np.asanyarray(np.isnan(df_mask)))

#     return x_miss, missing_indexes

# def get_schema(X_keys):
#     length = len(X_keys)
#     attrs = ['categorical']*length
#     X_schema = pd.DataFrame({'attribute':X_keys, 'type':attrs})
#
#     return X_schema


def formalize_imputed_df(df_imputed, mapping, keys):
    for i, info in enumerate(mapping):
        if info['datatype']!='float32' and info['datatype']!='float64':
            df_imputed[keys[i]] = df_imputed[keys[i]].astype('int64').fillna(-1)

def formalize_df(acquired_df_miss, mapping, keys):
    for i, info in enumerate(mapping):
        if info['datatype']=='int64':
            acquired_df_miss[keys[i]] = acquired_df_miss[keys[i]].fillna(-1).astype('int64')
        elif info['datatype']=='float64':
            acquired_df_miss[keys[i]] = acquired_df_miss[keys[i]].fillna(-1)
        else:
            category_num = len(acquired_df_miss[keys[i]].cat.categories)
            acquired_df_miss[keys[i]] = acquired_df_miss[keys[i]].cat.add_categories(category_num)
            acquired_df_miss[keys[i]] = acquired_df_miss[keys[i]].fillna(category_num)

def transform_vector_to_df(np_array, mapping, keys):
    df_from_np = pd.DataFrame(np_array, columns=keys)
    for i, k in enumerate(keys):
        if mapping[i]['datatype']=='O':
            df_from_np[k] = df_from_np[k].astype('category')
        else:
            df_from_np[k] = df_from_np[k].astype('float64')

    return df_from_np

# Generate incomplete tables different patterns
# pattern is one of the integers of [0,1,2,3,4] 
def generate_incomplete_table(X, missing_rate, pattern=0):
    '''
    hyperparameters:
    observed_prob: how much is the rate of observed data 
    quantile: quantile level at which the cuts should occur
    '''  
    observed_prob = 0.5
    quantile = 0.3
    
    # different missing patterns
    if pattern==0:
        X_miss = produce_NA(X, p_miss=missing_rate, mecha="MCAR")
    elif pattern==1:
        X_miss = produce_NA(X, p_miss=missing_rate/(1-observed_prob), mecha="MAR", p_obs=observed_prob)
    elif pattern==2:
        X_miss = produce_NA(X, p_miss=missing_rate, mecha="MNAR", opt="logistic", p_obs=observed_prob)
    elif pattern==3:
        X_miss = produce_NA(X, p_miss=missing_rate, mecha="MNAR", opt="selfmasked")
    else:
        X_miss = produce_NA(X, p_miss=missing_rate/observed_prob*2, mecha="MNAR", opt="quantile", p_obs=observed_prob, q=quantile)
    
    return X_miss


# Query on student tables
# input three tables to join
def query_on_tables(fact, key, nde):
    join_table = pd.merge(pd.merge(fact, key, left_on='state_code', right_on='State_Code '), nde,
                          left_on='State', right_on='state')
    res_ground = join_table[
        (join_table['c25'] <= 50000) & (join_table['c25'] >= 30000) & (join_table['average_scale_score'] >= 282) &
        (~join_table['state_code'].isnull()) & (pd.notna(join_table['state']))]

    return res_ground

def query_on_df(qs, df):
    df_query = df
    for q in qs:
        if q['width']==0:
            df_query = df_query[df_query[q['col']]==q['center']]
        else:
            df_query = df_query[(df_query[q['col']]>=q['center']-q['width']/2)&(df_query[q['col']]<=q['center']+q['width']/2)]

    return df_query
    
# def candidates_tables(fact, key, nde):
#     join_table = pd.merge(pd.merge(fact, key, left_on='state_code', right_on='State_Code '), nde,
#                           left_on='State', right_on='state')
#     res_ground = join_table[
#         (join_table['c25'] <= 50000) & (join_table['c25'] >= 30000) & (join_table['average_scale_score'] >= 282)]
#
#     return res_ground

# for <NA>, pd.NA, use pd.notna()
# for NaN, np.nan, use .isnull()
# the comparison of NaN and NaN, <NA> and <NA> have been implemented.
# but np.nan can not compare with any number.
def candidates_tables(fact, key, nde):
    join_table = pd.merge(pd.merge(fact, key, left_on='state_code', right_on='State_Code '), nde,
                          left_on='State', right_on='state')
    res_ground = join_table[
        (((join_table['c25'] <= 50000) & (join_table['c25'] >= 30000)) | (join_table['c25'].isnull())) &
        ((join_table['average_scale_score'] >= 282) | (join_table['average_scale_score']).isnull())]

    return res_ground

# input: the indexes of all the missing cells
# output: a list of all the rows containing missing values
def get_candidate_rows(query_columns, miss_rows, miss_cols):
    candidate_rows = []
    for i in range(len(miss_rows)):
        if miss_cols[i] in query_columns:
            candidate_rows.append(miss_rows[i])
    candidate_rows = list(set(candidate_rows))
    
    return candidate_rows

# input: multiple imputors, a single query
# output: line numbers of the top-k cells to acquire with maximum utility
# input: multiple imputors, multiple queries
# output: line numbers of the top-k cells to acquire with maximum utility
def get_utility_score(kernel, queries, k=100):
    acquire_task = []
    for query in queries:
        for i in range(kernel.dataset_count()):
            acquire_task.extend(query_on_df(query, kernel.complete_data(i)).index.values.tolist())
    acquire_task.sort()
    counter = collections.Counter(acquire_task)
    # Get items with non-zero frequency
    acq_task = [(item, count) for item, count in counter.items() if count != 0]

    #acq_task = counter.most_common(k)
  
    return acq_task

def get_costs(df_needed_imputed):
    costs = []
    for i in range(len(df_needed_imputed)):
        l = len(np.where(df_needed_imputed.iloc[i].isna())[0])
        costs.append(l*(0.99**l))
    
    return costs

def get_uncertainty_score(kernel, query, candidate_rows):
    acquire_task = []
    for i in range(kernel.dataset_count()):
        acquire_task.extend(query_on_df(query, kernel.complete_data(i).iloc[candidate_rows]).index.values.tolist())
    acquire_task.sort()
    counter = collections.Counter(acquire_task)
    uncertainty_scores_dict = {}
    for row in candidate_rows:
        uncertainty_scores_dict[row] = counter[row]*(kernel.dataset_count()-counter[row])
    
    sorted_uncertainty_score_indexes = sorted(uncertainty_scores_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_uncertainty_score_indexes = [x[0] for x in sorted_uncertainty_score_indexes]
  
    return sorted_uncertainty_score_indexes

def generate_weights(n):
    # Generate 100 random integer numbers within the range 1 to 10
    # Set the parameters (alpha values) for the Dirichlet distribution
    alpha = np.random.randint(1, 11, size=n)

    # Generate a random sample from the Dirichlet distribution
    portion_distribution = np.random.dirichlet(alpha)

    return portion_distribution


def greedy_and_improve_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    _rows, _cols = np.where(np.isnan(df_miss.to_numpy()))
    budget_units = budget*df_miss.isna().sum().sum() # it is not a integer
    
    # Create kernels. 
    kernel = mf.ImputationKernel(
      data=df_miss_copy,
      datasets=5,
      save_all_iterations=False,
      random_state=1991,
    )

    # Run the MICE algorithm for 3 iterations on each of the datasets
    kernel.mice(3, verbose=True, min_data_in_leaf = 6)
    
    acq_task = get_utility_score(kernel, queries, int(budget_units/0.8))
    row_indexes = [t[0] for t in acq_task]
    df_needed_imputed = df_miss_copy.loc[row_indexes]
    costs = get_costs(df_needed_imputed)
    
    util_per_cost = [(acq_task[i][0], costs[i]/acq_task[i][1]) for i in range(len(costs))]
    sorted_util_per_cost = sorted(util_per_cost, key=lambda x: x[1])
    
    i = 0
    while budget_units>0:
        row_ind = sorted_util_per_cost[i][0]
        i+=1
        miss_cols = np.where(df_miss_copy.iloc[row_ind].isna())[0]
        for col in miss_cols:
            df_miss_copy.iat[row_ind,col] = df_arr.iat[row_ind,col]
        budget_units -= len(miss_cols)*(0.99**len(miss_cols))
    
    return df_miss_copy

def query_by_rounds(df_miss, df_arr, budgets, queries, W):
    df_miss_copy = df_miss.copy()
    for budget in budgets:
        df_miss_copy = greedy_and_improve_acquisition(df_miss_copy, df_arr, budget, queries, W)
    
    return df_miss_copy

def get_acc(queries, acquired_df_miss, df_arr, W):
    
    formalize_df(acquired_df_miss, mapping, df.keys().to_list())
    
    pres = 0
    rs = 0 
    # no difference in evaluation parts
    for i, query in tqdm(enumerate(queries)):
        # get the dataframe of acquired answer and ground truth
        data_product = query_on_df(query, acquired_df_miss)
        ground_truth = query_on_df(query, df_arr)

        # get precision
        ground_truth_indexes = set(ground_truth.index.values.tolist())
        data_product_indexes = set(data_product.index.values.tolist())
        intersect = ground_truth_indexes.intersection(data_product_indexes)
        precision = len(intersect)/len(ground_truth_indexes)

        # get rmse = precision * RMSE
        indexes_lst = list(intersect)
        imputing_error = get_avg_error(df_arr.iloc[indexes_lst], acquired_df_miss.iloc[indexes_lst], mapping)
        rmse = precision*(1-imputing_error)
        
        pres += precision*W[i]
        rs += rmse*W[i]
        
    return pres, rs
    
