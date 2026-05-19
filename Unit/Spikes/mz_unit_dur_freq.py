#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# __version_1__ = Michael Zimmerman #02.10.2022

# just a list of functions for my unit frequency and duration analysis

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import division
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy import stats
import scipy.stats as sstat
import scipy.signal as ssig
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA as sklearnPCA
# import h5py
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import fnmatch

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for "3_unit_dur_freq.ipynb" jupyter notebook
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def vis_resp_units(final_df, region='v1', time_window=[0.5, 0.7], z_score = 2):
    test_df = final_df[final_df['region'] == region]
    
    vis_resp = test_df[(test_df['times'] >= time_window[0]) 
                       & (test_df['times'] <= time_window[1]) 
                       & (test_df['zscore'] > z_score)]
    
    test2 = test_df[test_df.cuid.isin(vis_resp.cuid.unique())]
    return test2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# detect duration of persistent activity/oscillations after onset of tim for single unit
# returns absolute duration after stim onset in sec
# dor duration analysis units should satisfy the following: 1) 1st peak<0.7s, 
# 2) time btw peaks max = 0.5s (2Hz), 3)max duration less than 2s 
from detect_peaks import detect_peaks
def _duration_peaks_unit(unit):
    data = unit
    peakind = detect_peaks(data, mph=1.5, mpd=10)
    if peakind.size>0:
        if peakind.size>1:
            if peakind[0]>50 and peakind[0]<70:
                mask = np.array(np.diff(peakind)<50)
                for ind, val in enumerate(mask):
                    if val==False:
                        mask[ind:]=False
                mask = np.insert(mask, 0, True)
                peakind = peakind[mask]
                dur = (peakind[-1]/100) - 0.5
            else:   
                dur = float('nan')
        elif peakind[0]>50 and peakind[0]<70:
            dur = (peakind[0]/100) - 0.5
        else:
            dur = float('nan')
        if dur<0 or dur>2:
            dur = float('nan')
    else:
        dur = float('nan')
    return dur, peakind
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def get_dur_peaks(all_array, group, situation):
#     all_array = all_array[:,int(0.5*100):int(1.5*100)] #n*100 bc the main_df.times are at 0.01 spacing
#     the_peaks = np.array([])
#     the_units = np.array([])
#     for ii in range(all_array.shape[0]):
#         yy = all_array[ii]
#         std = np.std(yy)
#         thresh = std*1.7
#         if situation == 'unrew':
#             peaks, _ = ssig.find_peaks(yy, prominence=.2, width=3, distance=15, height=0)
#         if situation == 'novel':
#             peaks, _ = ssig.find_peaks(yy, prominence=.2, width=3, distance=15, height=0, threshold=0.03)
#         else:
#             peaks, _ = ssig.find_peaks(yy, height=thresh, distance=15, prominence=1) # width=3
#         peaks = peaks/100 + 0.5
#         the_peaks = np.concatenate([the_peaks, peaks])
#         units = np.full(peaks.shape[0], ii)
#         the_units = np.concatenate([the_units, units])
#     peaks_df = pd.DataFrame({'peak_time': the_peaks, 'unit_id': the_units}, columns=['peak_time','unit_id'])
#     peaks_df['group'] = group
#     return peaks_df, yy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def get_osc_dur(df, group):
#     osc_dur = []
#     units = []
#     for ii in df.unit_id.unique():
#         working = df[df['unit_id'] == ii]
#         max_time = working['peak_time'].values.max()
#         min_time = working['peak_time'].values.min()
#         duration = max_time - min_time
#         if duration>0:
#             osc_dur.append(duration)
#             units.append(ii)
#     osc_dur_df = pd.DataFrame(list(zip(osc_dur,units)), columns=['duration', 'unit_id'])
#     osc_dur_df['group'] = group
#     return osc_dur_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def get_freq_peaks(all_array, group, situation):
#     all_array = all_array[:,int(0.5*100):int(1.75*100)] #n*100 bc the main_df.times are at 0.01 spacing
#     the_peaks = np.array([])
#     the_units = np.array([])
#     for ii in range(all_array.shape[0]):
#         yy = all_array[ii]
#         if situation == 'unrew':
#             peaks, _ = ssig.find_peaks(yy, prominence=.2, width=3, distance=15, height=0)
#         if situation == 'novel':
#             peaks, _ = ssig.find_peaks(yy, prominence=.2, width=3, distance=15, height=0, threshold=0.03)
#         else:
#             peaks, _ = ssig.find_peaks(yy, prominence=.1, width=3, distance=15, height=0)
#         peaks = peaks/100 + 0.5
#         the_peaks = np.concatenate([the_peaks, peaks])
#         units = np.full(peaks.shape[0], ii)
#         the_units = np.concatenate([the_units, units])
#     peaks_df = pd.DataFrame({'peak_time': the_peaks, 'unit_id': the_units}, columns=['peak_time','unit_id'])
#     peaks_df['group'] = group
#     return peaks_df, yy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Hz_subfunction(timearr):
    mean_vals = []
    max_vals = []
    
    for idx in range(timearr.shape[0]):
        yy = timearr[idx]
        mean_Hz = np.mean(yy)
        max_Hz = np.max(yy)
        mean_vals.append(mean_Hz)
        max_vals.append(max_Hz)
    
    return mean_vals, max_vals

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Hz_interval_peaks(Hz_array, group):
    
    time1 = Hz_array[:,int(0.45*100):int(0.65*100)]
    time2 = Hz_array[:,int(0.73*100):int(0.83*100)]
    time3 = Hz_array[:,int(0.9*100):int(1.05*100)]
    time4 = Hz_array[:,int(1.12*100):int(1.32*100)]
    time5 = Hz_array[:,int(1.4*100):int(1.6*100)]

    mean1, max1 = Hz_subfunction(time1)
    mean2, max2 = Hz_subfunction(time2)
    mean3, max3 = Hz_subfunction(time3)
    mean4, max4 = Hz_subfunction(time4)
    mean5, max5 = Hz_subfunction(time5)
    
    df1 = pd.DataFrame(list(zip(mean1, max1)), columns=['mean_hz', 'max_hz'])
    df1['time_id'] = '1'
    df2 = pd.DataFrame(list(zip(mean2, max3)), columns=['mean_hz', 'max_hz'])
    df2['time_id'] = '2'
    df3 = pd.DataFrame(list(zip(mean3, max3)), columns=['mean_hz', 'max_hz'])
    df3['time_id'] = '3'
    df4 = pd.DataFrame(list(zip(mean4, max4)), columns=['mean_hz', 'max_hz'])
    df4['time_id'] = '4'
    df5 = pd.DataFrame(list(zip(mean5, max5)), columns=['mean_hz', 'max_hz'])
    df5['time_id'] = '5'
    
    combo_df = pd.concat([df1, df2, df3, df4, df5])
    combo_df['group'] = group
    
    return combo_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Hz_overall_stats(df):
    varA = df[df['group']=='A'].mean_hz.values
    varB = df[df['group']=='B'].mean_hz.values
    return stats.ttest_ind(varA, varB)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Hz_peaks_stats(df):
    stats_results = []
    for cycle in df.time_id.unique():
        foo_df = df[df['time_id'] == cycle]
        var1 = foo_df[foo_df['group'] == 'A'].mean_hz.values
        var2 = foo_df[foo_df['group'] == 'B'].mean_hz.values
        result = stats.ttest_ind(var1, var2)
        stats_results.append(cycle)
        stats_results.append(result)
    
    return stats_results

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_Hz_peaks(Hz_array, group):
    Hz_array2 = Hz_array[:,int(0*100):int(2.5*100)]
    Hz_peaks = np.array([])
    for idx in range(Hz_array2.shape[0]):
        yy = Hz_array2[idx]
        std = np.std(yy)
        thresh = std*2
        peaks, out_dict = ssig.find_peaks(yy, height=(thresh,20), prominence=1, width=3)
        foo = out_dict['peak_heights']
        Hz_peaks = np.concatenate([Hz_peaks, foo])
    peaks_df = pd.DataFrame(Hz_peaks, columns=['peak_Hz'])
    peaks_df['group'] = group
    
    return peaks_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for "4_unit_situation_preference.ipynb" jupyter notebook
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def unit_sorting(main_df, dependent_df, dependent2_df, dfs_array='dfs'):
    main = main_df.pivot(index='cuid', columns='times', values='ztc')
    dependent = dependent_df.pivot(index='cuid', columns='times', values='ztc')
    dependent2 = dependent2_df.pivot(index='cuid', columns='times', values='ztc')
    
#     print(main.head())
#     print(dependent.head())
#     print(dependent2.head())
    
    #these are the indices of the units sorted in the rew situation
    foo = np.argsort(np.mean(main.values[:,50:70], axis = 1) )
    sorted_cuid = main.index[foo].tolist()
        
    #this sorts the unrewarded situation to the same order
    sorted_main = main.reindex(sorted_cuid)
    sorted_dependent_df = dependent.reindex(sorted_cuid)
    sorted_dependent2_df = dependent2.reindex(sorted_cuid)
    
    #this is for plotting the values on the heatmap
    val_main = sorted_main.values
    val_dependent = sorted_dependent_df.values
    val_dependent2 = sorted_dependent2_df.values
    
    if dfs_array == 'dfs':
        return sorted_main, sorted_dependent2_df, sorted_dependent_df
    elif dfs_array == 'array':
        return val_main, val_dependent, val_dependent2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def unit_pref_idx(rew_arr, unrew_arr):
    go_axis = []
    nogo_axis = []
    osi_ls = []
    for unit in range(rew_arr.shape[0]):
        try:
            rew = np.array(rew_arr)[unit,50:70].max() #time window incorporated for finding the max Hz
        except:
            rew = 0 #0 if and only if the unit never fires for this stimulus
        try:
            unrew = np.array(unrew_arr)[unit,50:70].max() #time window incorporated for finding the max Hz
        except:
            unrew = 0 #0 if and only if the unit never fires for this stimulus
        go_axis.append(rew)
        nogo_axis.append(unrew)
        osi = (rew-unrew)/(rew+unrew)
        osi_ls.append(osi)
    return go_axis, nogo_axis, osi_ls

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def go_nogo_activity_plot(rew_array, unrew_array):
    go_axis = []
    nogo_axis = []
    for unit in range(rew_array.shape[0]):
        rew_val = np.array(rew_array)[unit].max()
        unrew_val = np.array(unrew_array)[unit].max()
        go_axis.append(rew_val)
        nogo_axis.append(unrew_val)

    go_axis = np.array(go_axis)
    nogo_axis = np.array(nogo_axis)
    
    return go_axis, nogo_axis

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




