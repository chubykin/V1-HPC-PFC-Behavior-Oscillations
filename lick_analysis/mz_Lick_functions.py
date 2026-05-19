#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# __version_1__ = Michael Zimmerman #12.07.2021

# This is a new .py file to house all of the functions I've created and use for lick analysis

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import division
import pickle
import re 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
import os
# import joypy

import scipy.stats as sstat
ZZ = sstat.norm.ppf


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def new_get_licks_df(path, overall_order, inter_dur):
    f=open(path, 'rb')
    objs = []
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    
    rew_time_ls, rew2_time_ls, unrew_time_ls, true_trials = new_fun_1(objs, overall_order)

    df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr = new_fun_2(objs,
                                                                           overall_order,
                                                                           rew_time_ls, 
                                                                           rew2_time_ls,
                                                                           unrew_time_ls,
                                                                           inter_dur)

    df_licks2 = new_fun_3(df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr, true_trials)
    
    return df_licks2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def new_fun_1(objs, overall_order):
    #load data and extract lick times
    rew_time_ls = []
    rew2_time_ls = []
    unrew_time_ls = []
    rew_trials = [] # trial number list
    rew2_trials = []
    unrew_trials = []
    for idx, x in enumerate(objs):
        if overall_order[idx] == '0':
            try:
                rew_time_ls.append(x.split("r")[1].split(" ")[0].split("u")[0])
                rew_trials.append(idx)
            except:
                rew_time_ls.append(x.split("m")[1].split(" ")[0].split("u")[0])
                rew_trials.append(idx)
        elif overall_order[idx] == '1':
            try:
                rew2_time_ls.append(x.split("z")[1].split(" ")[0].split("u")[0])
                rew2_trials.append(idx)
            except:
                try:
                    rew2_time_ls.append(x.split("r")[1].split(" ")[0].split("u")[0])
                    rew2_trials.append(idx)
                except:
                    rew2_time_ls.append(x.split("m")[1].split(" ")[0].split("u")[0])
                    rew2_trials.append(idx)
        elif overall_order[idx] == '2':
            try:
                unrew_time_ls.append(x.split("o")[1].split(" ")[0].split("u")[0])
                unrew_trials.append(idx)
            except:
                print("Error{0}".format(idx))
    true_trials = rew_trials+rew2_trials+unrew_trials # the order here is important
    return rew_time_ls, rew2_time_ls, unrew_time_ls, true_trials

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def new_fun_2(objs, overall_order, rew_time_ls, rew2_time_ls, unrew_time_ls, inter_dur):
    #converting the string to an array
    rew_time_ar = np.array(rew_time_ls, dtype=np.float32)
    rew2_time_ar = np.array(rew2_time_ls, dtype=np.float32)
    unrew_time_ar = np.array(unrew_time_ls, dtype=np.float32)

    #subtracting the stimulus duration
    rew_times = rew_time_ar - inter_dur
    rew2_times = rew2_time_ar - inter_dur
    unrew_times = unrew_time_ar - inter_dur

    #removing the 'u', 'r', 'z', 'o' and 'm' identifiers
    cleaned_objs = [ x.replace('u',' ').replace('r', ' ').replace('z', ' ').replace('o', ' ').replace('m', ' ') for x in objs ]
    
    rew_cleaned = []
    rew2_cleaned = []
    unrew_cleaned = []
    for idx,val in enumerate(overall_order):
        if val == '0':
            rew_cleaned.append(cleaned_objs[idx])
        elif val == '1':
            rew2_cleaned.append(cleaned_objs[idx])
        elif val == '2':
            unrew_cleaned.append(cleaned_objs[idx])

    rew_cleaned = np.array(rew_cleaned)
    rew2_cleaned = np.array(rew2_cleaned)
    unrew_cleaned = np.array(unrew_cleaned)
    
    rew_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in rew_cleaned ]
    rew2_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in rew2_cleaned ]
    unrew_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in unrew_cleaned ]

    for idx, val in enumerate(rew_time_ar):
        rew_lick[idx] =  rew_lick[idx][rew_lick[idx] != val]
    for idx, val in enumerate(rew2_time_ar):
        rew2_lick[idx] =  rew2_lick[idx][rew2_lick[idx] != val]
    for idx, val in enumerate(unrew_time_ar):
        unrew_lick[idx] =  unrew_lick[idx][unrew_lick[idx] != val]
    
    # Create data frame for the rewarded - water - trials
    rew_df = pd.DataFrame(rew_lick).sub(rew_times, axis = 0)
    num_rew_tr = rew_df.shape[0]
    # Create data frame for the rewarded - no water - trials
    rew2_df = pd.DataFrame(rew2_lick).sub(rew2_times, axis = 0)
    num_rew2_tr = rew2_df.shape[0]
    # Create data frame for the unrewarded trials
    unrew_df = pd.DataFrame(unrew_lick).sub(unrew_times, axis = 0)
    num_unrew_tr = unrew_df.shape[0]
    
    df_licks = pd.concat([rew_df, rew2_df, unrew_df], ignore_index=True) # the order here is important
    df_licks.drop(df_licks.index[0], inplace=True)
    df_licks2 = pd.DataFrame(df_licks.stack())
    
    return df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def new_fun_3(df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr, true_trials):
    test = df_licks.to_numpy()

    list_of_licks_per_trial = []
    counter = 0
    for i in range(num_rew_tr+num_rew2_tr+num_unrew_tr-1):
        for value in test[i]:
            if value > -100000000 and value < 10000000:
                counter += 1                    # update the counter by one for each lick during a trial
            else:
                break
        list_of_licks_per_trial.append(counter) # this created a list with the # of licks per trial
        counter = 0                             # resets the counter for each trial

    list_reward_and_none = []
    for i, val in enumerate(list_of_licks_per_trial):
        for j in range(val):
            if i < num_rew_tr:       # Check if it's less than or less than or equal
                list_reward_and_none.append("rew")
            elif i > num_rew_tr and i < (num_rew_tr+num_rew2_tr):
                list_reward_and_none.append("rew2")
            else:
                list_reward_and_none.append("unrew")

    df_licks2 = df_licks2.reset_index()
    df_licks2.columns = ["trial", "lick_idx", "lick_time"]
    df_licks2["stim_id"] = list_reward_and_none

    trial_ls=list(range(0,150))
    map_dict=dict(zip(trial_ls,true_trials))
    df_licks2['true_tr'] = df_licks2.trial.map(map_dict)
    return df_licks2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_training_score(df, if_print=1):
    rewarded_trials = 95
    unrew_trials = 50
    
    results = df.groupby('stim_id')['trial'].nunique()
    
    try:
        num_tr_rew = results['rew']
    except:
        num_tr_rew = 0
    try:
        num_tr_rew2 = results['rew2']
    except:
        num_tr_rew2 = 0
    try:
        num_tr_unrew = results['unrew']
    except:
        num_tr_unrew = 0
    
    num_tr_rew_rew2 = num_tr_rew + num_tr_rew2
    
    Hits_rate = num_tr_rew_rew2/rewarded_trials
    if Hits_rate >= 1:
        Hits_rate = 0.99999
    elif Hits_rate == 0:
        Hits_rate = 0.00001
        
    FA_rate = num_tr_unrew/unrew_trials
    if FA_rate >= 1:
        FA_rate = 0.99999
    elif FA_rate == 0:
        FA_rate = 0.00001

    Training_Score = ZZ(Hits_rate) - ZZ(FA_rate)
#     Training_Score = (num_tr_rew_rew2-num_tr_unrew)/(rewarded_trials+unrew_trials)

#     if np.isnan(Training_Score):
#         Training_Score = 0
    
    if if_print == 1:
        print('Hits: {0} ----- FA: {1}'.format(Hits_rate, FA_rate))
    
    return Training_Score

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ls_to_dict(a): #converting the overall_ls to a dictionary
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_group_label(key):
    global pc
    key_cc = key.split("_")[0]
    if (key_cc == "CC082263") | (key_cc == "CC067489") | (key_cc == "CC082260") | (key_cc == "CC084621"):
        pc = 'cyan'
        print('WT')
    elif (key_cc == "CC082257") | (key_cc == "CC067431") | (key_cc == "CC067432") | (key_cc == "CC082255"):
        pc = 'magenta'
        print('FX')
    return pc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Below are the old, outdated versions of the functions updated above - MZ 12.07.2021

# def get_licks_df(path):
#     f=open(path, 'rb')
#     objs = []
#     while 1:
#         try:
#             objs.append(pickle.load(f))
#         except EOFError:
#             break
#     f.close()
    
    
#     rew_time_ls, rew2_time_ls, unrew_time_ls, miss_ls = fun_1(objs)

#     df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr, num_miss_tr = fun_2(objs,
#                                                                                     rew_time_ls, 
#                                                                                     rew2_time_ls,
#                                                                                     unrew_time_ls, 
#                                                                                     miss_ls)

#     df_licks2 = fun_3(df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr, num_miss_tr)
    
#     return df_licks2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def fun_1(objs):
#     #load data and extract lick times
#     rew_time_ls = []
#     rew2_time_ls = []
#     unrew_time_ls = []
#     miss_ls= []
#     for idx, x in enumerate(objs):
#         if overall_order[idx] == '0':
#             try:
#                 rew_time_ls.append(x.split("r")[1].split(" ")[0].split("u")[0])
#             except:
#                 miss_ls.append(x.split("m")[1].split(" ")[0].split("u")[0])
#         elif overall_order[idx] == '1':
#             try:
#                 rew2_time_ls.append(x.split("z")[1].split(" ")[0].split("u")[0])
#             except:
#                 try:
#                     rew2_time_ls.append(x.split("r")[1].split(" ")[0].split("u")[0]) #older training days didn't have "z"
#                 except:
#                     miss_ls.append(x.split("m")[1].split(" ")[0].split("u")[0])            
#         elif overall_order[idx] == '2':
#             try:
#                 unrew_time_ls.append(x.split("o")[1].split(" ")[0].split("u")[0])
#             except:
#                 print("Error{0}".format(idx))

# #     print(len(rew_time_ls))
# #     print(len(rew2_time_ls))
# #     print(len(unrew_time_ls))
# #     print(len(miss_ls))
    
# #     total_trials = len(rew_time_ls)+len(rew2_time_ls)+len(unrew_time_ls)+len(miss_ls)
# #     print(total_trials)
    
#     return rew_time_ls, rew2_time_ls, unrew_time_ls, miss_ls

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def fun_2(objs, rew_time_ls, rew2_time_ls, unrew_time_ls, miss_ls):
#     #converting the string to an array
#     rew_time_ar = np.array(rew_time_ls, dtype=np.float32)
#     rew2_time_ar = np.array(rew2_time_ls, dtype=np.float32)
#     unrew_time_ar = np.array(unrew_time_ls, dtype=np.float32)
#     miss_time_ar = np.array(miss_ls, dtype=np.float32)

#     #subtracting the stimulus duration
#     rew_times = rew_time_ar - inter_dur
#     rew2_times = rew2_time_ar - inter_dur
#     unrew_times = unrew_time_ar - inter_dur
#     miss_times = miss_time_ar - inter_dur

#     #removing the 'u', 'r', 'z', 'o' and 'm' identifiers
#     cleaned_objs = [ x.replace('u', ' ') for x in objs if 'u' in x ]

#     rew_cleaned = [ x.replace('r',' ') for x in cleaned_objs if 'r' in x ]
#     rew2_cleaned = [ x.replace('z',' ') for x in cleaned_objs if 'z' in x ]
#     unrew_cleaned = [ x.replace('o',' ') for x in cleaned_objs if 'o' in x ]
#     miss_cleaned = [ x.replace('m',' ') for x in cleaned_objs if 'm' in x]
    
#     rew_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in rew_cleaned ]
#     rew2_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in rew2_cleaned ]
#     unrew_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in unrew_cleaned ]
#     miss_lick = [np.array(x.strip().split(" "), dtype=np.float32) for x in miss_cleaned ]

#     for idx, val in enumerate(rew_time_ar):
#         rew_lick[idx] =  rew_lick[idx][rew_lick[idx] != val]
#     for idx, val in enumerate(rew2_time_ar):
#         rew2_lick[idx] =  rew2_lick[idx][rew2_lick[idx] != val]
#     for idx, val in enumerate(unrew_time_ar):
#         unrew_lick[idx] =  unrew_lick[idx][unrew_lick[idx] != val]
#     for idx, val in enumerate(miss_time_ar):
#         miss_lick[idx] = miss_lick[idx][miss_lick[idx] != val]
        
#     # Create data frame for the rewarded - water - trials
#     rew_df = pd.DataFrame(rew_lick).sub(rew_times, axis = 0)
#     num_rew_tr = rew_df.shape[0]
#     # Create data frame for the rewarded - no water - trials
#     rew2_df = pd.DataFrame(rew2_lick).sub(rew2_times, axis = 0)
#     num_rew2_tr = rew2_df.shape[0]
#     # Create data frame for the unrewarded trials
#     unrew_df = pd.DataFrame(unrew_lick).sub(unrew_times, axis = 0)
#     num_unrew_tr = unrew_df.shape[0]
#     # Create a data frame for the rewarded trials that had no licks during stim time
#     miss_df = pd.DataFrame(miss_lick).sub(miss_times, axis = 0)
#     num_miss_tr = miss_df.shape[0]

#     df_licks = pd.concat([rew_df, rew2_df, miss_df, unrew_df], ignore_index=True) # the order here is important
#     df_licks.drop(df_licks.index[0], inplace=True)
#     df_licks2 = pd.DataFrame(df_licks.stack())
    
#     return df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr, num_miss_tr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def fun_3(df_licks, df_licks2, num_rew_tr, num_rew2_tr, num_unrew_tr, num_miss_tr):
#     test = df_licks.to_numpy()

#     list_of_licks_per_trial = []
#     counter = 0
#     for i in range(num_rew_tr+num_rew2_tr+num_unrew_tr+num_miss_tr-1):
#         for value in test[i]:
#             if value > -100000000 and value < 10000000:
#                 counter += 1                    # update the counter by one for each lick during a trial
#             else:
#                 break
#         list_of_licks_per_trial.append(counter) # this created a list with the # of licks per trial
#         counter = 0                             # resets the counter for each trial


#     list_reward_and_none = []
#     for i, val in enumerate(list_of_licks_per_trial):
#         for j in range(val):
#             if i < num_rew_tr:       # Check if it's less than or less than or equal
#                 list_reward_and_none.append("rew")
#             elif i > num_rew_tr and i < (num_rew_tr+num_rew2_tr):
#                 list_reward_and_none.append("rew2")
#             elif i > (num_rew_tr+num_rew2_tr) and i < (num_rew_tr+num_rew2_tr+num_miss_tr):
#                 list_reward_and_none.append("miss")
#             else:
#                 list_reward_and_none.append("unrew")
    
#     df_licks2 = df_licks2.reset_index()
#     df_licks2.columns = ["trial", "lick_idx", "lick_time"]

#     df_licks2["stim_id"] = list_reward_and_none
    
# #     print(df_licks2["stim id"].unique())
    
#     return df_licks2