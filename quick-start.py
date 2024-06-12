#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
import collections
import json

import numpy as np
import pandas as pd
import pandasql as pdsql
import miceforest as mf
from warnings import simplefilter
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from utils.util_preprocess import *


# Global variables for CDA

# Load dataset
# 1. Initial dataset:   
path_dataset = "./data/"

# preprocess a dataset
# check the initialized schema manually first.
df = pd.read_csv(path_dataset + "forest10/original.csv")

# df_arr original table to vectors
#         integer remains int64
#         float remains float64
#         others to category
# mapping schema
df_arr, mapping = df_2_array(df)
# df_arr_new, mapping = df_2_array(df)


# 2.EXP settings
missing_rate = 0.3
pattern = 0
budget = 0.05

multiple_imputing_num = 5 # number of models for multiple imputing


# ## 3. Incomplete Dataset
# ----> df_miss, x_miss, miss_rows, miss_cols
# do not deep copy df_miss or x_miss
# df_miss have the same schema with df_arr
#         but all numbers are float64
# df_miss and x_miss do not share the same part of memory
df_miss = mf.ampute_data(df_arr, perc=missing_rate, random_state=1991)
# df_miss = mf.ampute_data(df_arr, variables=[0, 1, 2, 9], perc=missing_rate, random_state=1991)
# x_miss = df_miss.to_numpy()
# miss_rows, miss_cols = np.where(np.isnan(x_miss))


# --> queries
queries = []
with open(path_dataset + "forest10/queries.json") as f:
    for eachline in f:
        query = json.loads(eachline.strip('\n'))
        queries.append(query['data'])
W = generate_weights(len(queries))


# Real Implementation
def query_on_df(qs, df):
    df_query = df
    for q in qs:
        if q['width']==0:
            df_query = df_query[df_query[q['col']]==q['center']]
        else:
            df_query = df_query[(df_query[q['col']]>=q['center']-q['width']/2)&(df_query[q['col']]<=q['center']+q['width']/2)]

    return df_query

def get_uncertainty_score(kernel, queries, k=100):
    acquire_task = []
    for query in queries:
        acquire_task_freq = []
        for i in range(kernel.dataset_count()):
            acquire_task_freq.extend(query_on_df(query, kernel.complete_data(i)).index.values.tolist())
        acquire_task_freq.sort()
        counter = collections.Counter(acquire_task_freq)
        
        uncertainty_N = kernel.dataset_count()
        acq_task = [(item, count*(uncertainty_N-count)) for item, count in counter.items() if count!=0]
        
        acquire_task.extend(acq_task)
    
    freq_dict = {}
    for t in acquire_task:
        if t[0] not in freq_dict:
            freq_dict[t[0]] = 0
        freq_dict[t[0]] += t[1]
    
    #acq_task = counter.most_common(k)
    return list(freq_dict.items())

def uncertainty_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    budget_units = budget*df_miss.isna().sum().sum() # it is not a integer
    
    # Create kernels. 
    kernel = mf.ImputationKernel(
      data=df_miss_copy,
      datasets=10,
      save_all_iterations=False,
      random_state=1991,
    )

    # Run the MICE algorithm for 3 iterations on each of the datasets
    kernel.mice(3, verbose=True, min_data_in_leaf = 6)

    
    acq_task = get_uncertainty_score(kernel, queries, int(budget_units/0.8))
    row_indexes = [t[0] for t in acq_task]
    df_needed_imputed = df_miss_copy.loc[row_indexes]
    costs = get_costs(df_needed_imputed)
    
    util_per_cost = [(acq_task[i][0], costs[i]/acq_task[i][1]) for i in range(len(costs)) if acq_task[i][1]!=0]
    sorted_util_per_cost = sorted(util_per_cost, key=lambda x: x[1])
    
    additional_indexes = random.sample(list(set(list(df_arr.index))-set(row_indexes)), int(budget_units))
    additional_util_per_cost = [(_, 120000) for _ in additional_indexes]
    sorted_util_per_cost.extend(additional_util_per_cost)

    i = 0       
    while budget_units>0:
        if i>=len(sorted_util_per_cost):
            break
        row_ind = sorted_util_per_cost[i][0]
        i+=1
        miss_cols = np.where(df_miss_copy.iloc[row_ind].isna())[0]
        for col in miss_cols:
            if col in [0, 1, 2, 9]:
                df_miss_copy.iat[row_ind,col] = df_arr.iat[row_ind,col]
                budget_units -= 1
    
    return df_miss_copy

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

def greedy_and_improve_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    #_rows, _cols = np.where(np.isnan(df_miss.to_numpy()))
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
    
    util_per_cost = [(acq_task[i][0], costs[i]/acq_task[i][1]) for i in range(len(costs)) if acq_task[i][1]!=0]
    sorted_util_per_cost = sorted(util_per_cost, key=lambda x: x[1])
    
    additional_indexes = random.sample(list(set(list(df_arr.index))-set(row_indexes)), int(budget_units))
    additional_util_per_cost = [(_, 120000) for _ in additional_indexes]
    sorted_util_per_cost.extend(additional_util_per_cost)

    i = 0
    while budget_units>0:
        if i>=len(sorted_util_per_cost):
            break
        row_ind = sorted_util_per_cost[i][0]
        i+=1
        miss_cols = np.where(df_miss_copy.iloc[row_ind].isna())[0]
        for col in miss_cols:
            #if col in [0, 1, 2, 9]:
            df_miss_copy.iat[row_ind,col] = df_arr.iat[row_ind,col]
            budget_units -= 1
    
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
        if len(ground_truth)==0:
            precision = 1
            rmse = 1
        else:
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



def main():
    st = time.time()
    acquired_df_miss = query_by_rounds(df_miss, df_arr, [budget/3, budget/3, budget/3], queries, W)
    et = time.time()

    pres_original, rs_original = get_acc(queries, df_miss, df_arr, W)
    pres, rs = get_acc(queries, acquired_df_miss, df_arr, W)  

    print("Without CDA, the recall is "+ str(pres_original))
    print("With CDA, the recall is "+ str(pres))


if __name__ == '__main__':
    main()





