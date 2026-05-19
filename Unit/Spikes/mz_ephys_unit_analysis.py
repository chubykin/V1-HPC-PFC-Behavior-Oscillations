#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# __version_1__ = Michael Zimmerman #06.30.2021

# This is primarily copied from the lab's ephys_unit_analysis.py file
#      - I did this to work with Neuropixels probes while using the same functions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import division
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import scipy.stats as sstat
import scipy.signal as ssig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
import h5py
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import fnmatch

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def PSTH(times, th_bin, trace_length, trials_number):
    #modified psth function from Dr Chubykin
    #output is one column vector which is mean psth across all trials
    #trials number given as 1 because times are rescaled in getRaster function

    trace_length_all=trace_length
    tht=np.arange(0,trace_length_all,th_bin)
    sigma=0.05 #sd of kernel in ms
    edges=np.arange(-2*sigma,2*sigma,th_bin) # 100 ms from both side, big variance

    edges_count=np.size(edges)
    center = int(np.ceil(edges_count/2))
    #yy=scipy.signal.general_gaussian(edges_count,0,1/sigma)
    yy = ssig.gaussian(edges_count,1/(4*sigma)) # the integral of the gaussian has to be 1 (trapz(yy)=1) - so renormalize 03/28/12
    yy = yy/np.trapz(yy) #integrate trapezoid/ normalize

    times = times # do need to compute times per trials because times are already rescaled in getRaster function

    trxlc=np.histogram(times,bins=tht)
    #print trxlc
    #trxlc=np.histogram(times,bins=trace_length/th_bin*trials_number)

    l2=np.array(trxlc[0])/trials_number # because we have twenty trials, we need to compute mean psth

    a1=np.convolve(yy,l2) #to smoothen histogram, reduces issues with spikes at edges of bin window

    a1=a1[center:int(trace_length_all/th_bin+center)] #take out only middle part 

    a1=a1/th_bin #compute the frequency rate

    #print l2.max()

    ttr=np.arange(0,np.size(a1)*th_bin,th_bin)

    #plt.plot(ttr,a1,'g-')
    """    
    plt.figure()
    plt.plot(ttr,a1,'g-')
    plt.suptitle('convolved, centered')
    """

    #trxlc=a1

    #n1=np.fix(trace_length/th_bin)
    #n1=np.fix(trace_length/th_bin)
    #a2=np.reshape(a1,(n1,trials_number), order='F') # Fortran order (column-major)
    #ttr2=np.arange(0,np.shape(a2)[0]*th_bin,th_bin)
    #print 'a2 shape:',np.shape(a2),' ttr2 shape:',np.shape(ttr2)
    #plt.figure()
    #plt.plot(ttr2,a2)    

    # Normalization of the returned PSTH to the spikes/s:
    #a2=a2*th_bin/1000
    return a1, ttr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getRaster_kilosort(df, cluster_number, trial_length): 
    cluster = df[df['cuid'] == cluster_number] 

    df = pd.DataFrame() 
    df['trial'] = (cluster.times//trial_length) 
    df['times'] =  cluster.times.sub((df.trial * trial_length), axis=0 ) 
    df.index = df.trial.values 

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_hdf(path):
    store = pd.HDFStore(path)
    ls_psth = []
    ls_spikes = []
    ls_wv = []
    d = {}
    for k in store.keys():
        if 'psth' in k:
            ls_psth.append(store[k])
        elif 'spikes' in k:
            ls_spikes.append(store[k])
        elif 'tmt' in k:
            ls_wv.append(store[k])
    try:

        d['spikes'] = pd.concat(ls_spikes)
        d['psth'] = pd.concat(ls_psth)
        d['tmt'] = pd.concat(ls_wv)
    except:
        print ('smth not loaded')
    store.close()
    return d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_mat(path):
    d = {}
    f = h5py.File(path)
    ar = np.array(f['rez']['st3']).T
    tmt_arr =  np.array(f['rez']['dWU'])
    d['tmt'] = tmt_arr
    df = pd.DataFrame(ar, columns=['samples', 'spike_templates', 'a', 'b', 'cuid'])
    df['times'] = df.samples/30000.0

    return df, d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _ztc(data): 
    df = data
    ls_tmp = []
    for i in df['cuid'].unique():

        tmp = df[df['cuid']==i]
        mean = tmp.loc[np.arange(0,50)].Hz.mean()
        std = tmp.loc[np.arange(0,50)].Hz.std()  

        mean2 = tmp.Hz.mean()
        std2 = tmp.Hz.std()  
        if mean==0:
            std =1    
        if mean2==0:
            std2 =1 

        tmp['ztc'] = (tmp.Hz - mean)/std 
        tmp['zscore'] = (tmp.Hz - mean2)/std2 

        ls_tmp.append(tmp) 

    df2 = pd.concat(ls_tmp)
    df2.dropna()
    df2.head()
    return df2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# maps group assignment(found by K-Means) to CUID 
def map_cluster(y_kmeans, n):
    ls = list(y_kmeans)
    d = {}
    j = int(0)
    for i in n:
        d[i]=ls[j]
        j = int(j + 1)
    return d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def unit_kmeans(data, n, key, t0, t1):
    tmp = data
    df_new = tmp.pivot(index='times', columns='cuid', values='zscore')

#     df_new = df_new.reset_index().drop('times',1)
    df_new = df_new.T
    
    X = df_new.iloc[:,t0:t1].values #0.5-2 second interval
    y = df_new.index.values.tolist() # corresponding cuid
    
    
    sklearn_pca = sklearnPCA(n_components=n) #compute 3 pc
    Y_sklearn = sklearn_pca.fit_transform(X)
    pca = sklearn_pca.fit(X)
    
    print(sklearn_pca.explained_variance_ratio_) 
    
    #pca = PCA(n_components=n_digits).fit(data)
    model = KMeans(init=pca.components_, n_clusters=n, n_init=1)
    
#     model = KMeans(n, init='k-means++', n_init=50, max_iter=100, tol=0.00001, 
#                    precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)  # 4 clusters
    model.fit(X)
    y_kmeans = model.predict(X)
    
    unique_id = df_new.index.values
    d = map_cluster(y_kmeans, unique_id)
    tmp[str(key)] = tmp['cuid'].map(d)
    tmp.describe()
#     fig = plt.figure(1, figsize=(8, 6))
#     ax = Axes3D(fig, elev=-150, azim=110)

#     ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], Y_sklearn[:, 2], c=y_kmeans,   cmap='rainbow')

#     ax.set_title("First three PCA directions")
#     ax.set_xlabel("1st eigenvector")
#     ax.w_xaxis.set_ticklabels([])
#     ax.set_ylabel("2nd eigenvector")
#     ax.w_yaxis.set_ticklabels([])
#     ax.set_zlabel("3rd eigenvector")
#     ax.w_zaxis.set_ticklabels([])
    return tmp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pca_kmeans(data, n, key):
    tmp = data
    df_new = tmp.pivot(index='times', columns='cuid', values='zscore')

    df_new = df_new.reset_index().drop('times',1)
    df_new = df_new.T
    X = df_new.iloc[:,50:250].values #osc interval
    y = df_new.index.values.tolist() # corresponding cuid

    sklearn_pca = sklearnPCA(n_components=n) #compute 3 pc
    Y_sklearn = sklearn_pca.fit_transform(X)
    print(sklearn_pca.explained_variance_ratio_) 
    
    model = KMeans(n)  # 4 clusters
    model.fit(X)
    y_kmeans = model.predict(X)

    #####For plotting scatter#######
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], Y_sklearn[:, 2], c=y_kmeans,   cmap='rainbow')

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector",fontsize = 20)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector",fontsize = 20)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector",fontsize = 20)
    ax.w_zaxis.set_ticklabels([])
    #####For plotting scatter####### 
    
    unique_id = df_new.index.values
    d = map_cluster(y_kmeans, unique_id)
    tmp[str(key)] = tmp['cuid'].map(d)
    tmp.describe()
    return tmp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def res_wilcox(data):
    cue_res = {}
    a = data.loc[np.arange(0,10)].groupby('clusterID') #0-50 corrsponds to 0 - 0.5 sec

    b = data.loc[np.arange(30,40)].groupby('clusterID')

    base = a.apply(lambda x: x.Hz.tolist())
    cue = b.apply(lambda x: x.Hz.tolist())
    for i in range(len(b.clusterID.unique())):

        w = sstat.wilcoxon(base.iloc[i], cue.iloc[i])
        if w[1] < 0.05:
            if (np.mean(cue.iloc[i]) - np.mean(base.iloc[i])) > 0:
                cue_res[base.index[i]] = 'exc'
            else:
                cue_res[base.index[i]] = 'inh'
        else:
            cue_res[base.index[i]] = 'ns'
    data['u_res'] = data['clusterID'].map(cue_res)
    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def calc_duration_zscore(self):
    dat = self[self.index<200] # first 2 sec of data

    x = []

    for jj in dat.cuid.unique():
        base = dat[  (dat.cuid ==jj) & (dat.index<50)].ztc
        d = dat[dat.cuid==jj].ztc
        thres = base.std()*2

        try:
            x.append(d[d>thres].index[-1]/100.0) 
        except:
            continue
    return x

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# detect duration of persistent activity/oscillations after onset of tim for single unit
# returns absolute duration after stim onset in sec
# dor duration analysis units should satisfy the following: 1) 1st peak<0.7s, 
# 2) time btw peaks max = 0.5s (2Hz), 3)max duration less than 2s 

def _duration_peaks_unit(unit):
    data = unit
    #data = unit.zscore
    #     data = data[:250]
    #thres = np.std(data[:50])

    #filt = scipy.signal.medfilt(data, 3)

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

    return dur
    #     plt.plot(data)
    #     plt.plot(idx, data[idx], 'ro')
    #     plt.plot(peakind, data[peakind], 'go')
    #     plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ksort_get_tmt(data, unit, templates, channel_groups):
    tmt_id = data[data.cluster_id==unit].templates.unique().tolist()
    tmt_arr = templates[tmt_id]
    tmt_arr = np.mean(tmt_arr, axis=0)
    ch_idx = tmt_arr.min(axis=0).argmin()
    depth = channel_groups['geometry'][ch_idx][1]
    tmt_avg = tmt_arr[:,ch_idx]
    return tmt_avg, depth, ch_idx
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def extract_interval(fname):
    #domain = re.search("[\w.+] Interval", fname)
    domain = re.search("[0-9.]+s [Ii]nterval", fname)
    if domain!=None:
        #print domain.group()
        #print domain.group()[:-10]
        inter_stim_interval=float(domain.group()[:-10])
    else:
        inter_stim_interval=None
    return inter_stim_interval 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### compute the area under the curve of z scores 
def auc(df, a, b, val):
    ls = []
    for i in df.cuid.unique():

        dat = df[df.cuid==i].val
    #     thres =  dat.ix[np.arange(200,250)].median() + dat.ix[np.arange(200,250)].std()*1.5
    #     print dat.ix[np.arange(200,250)].mean(), dat.ix[np.arange(200,250)].std()*2
    #     bl =  pd.rolling_mean(dat.ix[np.arange(50,200)], 10) > thres
    #     d = pd.DataFrame(bl)
    #     print bl.values
    #     dur = d[d.zscore==True].index.max()
        area = np.trapz(dat.ix[np.arange(a,b)][dat.ix[np.arange(a,b)] > 0])
    #     print area
        if area>0:
            ls.append(area)
    return ls

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_channel_depth(probe):
    if probe == '64DA':
        channel_groups = {
            'geometry': {
                        0: (0, 975),
                        1: (0, 875),
                        2: (0, 775),
                        3: (0, 675),
                        4: (0, 575),
                        5: (0, 475),
                        6: (0, 375),
                        7: (0, 275),
                        8: (0, 175),
                        9: (0, 75),
                        10: (0, 0),
                        11: (16, 50),
                        12: (20, 100),
                        13: (20, 150),
                        14: (20, 200),
                        15: (20, 250),
                        16: (20, 300),
                        17: (20, 1050),
                        18: (20, 1000),
                        19: (20, 950),
                        20: (20, 900),
                        21: (20, 850),
                        22: (20, 800),
                        23: (20, 750),
                        24: (20, 700),
                        25: (20, 650),
                        26: (20, 600),
                        27: (20, 550),
                        28: (20, 500),
                        29: (20, 450),
                        30: (20, 400),
                        31: (20, 350),
                        32: (-20, 300),
                        33: (-20, 350),
                        34: (-20, 400),
                        35: (-20, 450),
                        36: (-20, 500),
                        37: (-20, 550),
                        38: (-20, 600),
                        39: (-20, 650),
                        40: (-20, 700),
                        41: (-20, 750),
                        42: (-20, 800),
                        43: (-20, 850),
                        44: (-20, 900),
                        45: (-20, 950),
                        46: (-20, 1000),
                        47: (-20, 1050),
                        48: (-20, 250),
                        49: (-20, 200),
                        50: (-20, 150),
                        51: (-20, 100),
                        52: (-16, 50),
                        53: (0, 25),
                        54: (0, 125),
                        55: (0, 225),
                        56: (0, 325),
                        57: (0, 425),
                        58: (0, 525),
                        59: (0, 625),
                        60: (0, 725),
                        61: (0, 825),
                        62: (0, 925),
                        63: (0, 1025),
                        }
            }
    # probe 64DB, front
    if probe == '64DB':
        channel_groups = {
            'geometry': {
                        0: (0, 1025),
                        1: (0, 925),
                        2: (0, 825),
                        3: (0, 725),
                        4: (0, 625),
                        5: (0, 525),
                        6: (0, 425),
                        7: (0, 325),
                        8: (0, 225),
                        9: (0, 125),
                        10: (0, 25),
                        11: (-16, 50),
                        12: (-20, 100),
                        13: (-20, 150),
                        14: (-20, 200),
                        15: (-20, 250),
                        16: (-20, 1050),
                        17: (-20, 1000),
                        18: (-20, 950),
                        19: (-20, 900),
                        20: (-20, 850),
                        21: (-20, 800),
                        22: (-20, 750),
                        23: (-20, 700),
                        24: (-20, 650),
                        25: (-20, 600),
                        26: (-20, 550),
                        27: (-20, 500),
                        28: (-20, 450),
                        29: (-20, 400),
                        30: (-20, 350),
                        31: (-20, 300),
                        32: (20, 350),
                        33: (20, 400),
                        34: (20, 450),
                        35: (20, 500),
                        36: (20, 550),
                        37: (20, 600),
                        38: (20, 650),
                        39: (20, 700),
                        40: (20, 750),
                        41: (20, 800),
                        42: (20, 850),
                        43: (20, 900),
                        44: (20, 950),
                        45: (20, 1000),
                        46: (20, 1050),
                        47: (20, 300),
                        48: (20, 250),
                        49: (20, 200),
                        50: (20, 150),
                        51: (20, 100),
                        52: (16, 50),
                        53: (0, 0),
                        54: (0, 75),
                        55: (0, 175),
                        56: (0, 275),
                        57: (0, 375),
                        58: (0, 475),
                        59: (0, 575),
                        60: (0, 675),
                        61: (0, 775),
                        62: (0, 875),
                        63: (0, 975),
                        }
            }
    if probe == 'Neuropixels':
        channel_groups = {
            'geometry': {
                        0: (43, 20),
                        1: (11, 20),
                        2: (59, 40),
                        3: (27, 40),
                        4: (43, 60),
                        5: (11, 60),
                        6: (59, 80),
                        7: (27, 80),
                        8: (43, 100),
                        9: (11, 100),
                        10: (59, 120),
                        11: (27, 120),
                        12: (43, 140),
                        13: (11, 140),
                        14: (59, 160),
                        15: (27, 160),
                        16: (43, 180),
                        17: (11, 180),
                        18: (59, 200),
                        19: (27, 200),
                        20: (43, 220),
                        21: (11, 220),
                        22: (59, 240),
                        23: (27, 240),
                        24: (43, 260),
                        25: (11, 260),
                        26: (59, 280),
                        27: (27, 280),
                        28: (43, 300),
                        29: (11, 300),
                        30: (59, 320),
                        31: (27, 320),
                        32: (43, 340),
                        33: (11, 340),
                        34: (59, 360),
                        35: (27, 360),
                        36: (43, 380),
                        37: (11, 380),
                        38: (59, 400),
                        39: (27, 400),
                        40: (43, 420),
                        41: (11, 420),
                        42: (59, 440),
                        43: (27, 440),
                        44: (43, 460),
                        45: (11, 460),
                        46: (59, 480),
                        47: (27, 480),
                        48: (43, 500),
                        49: (11, 500),
                        50: (59, 520),
                        51: (27, 520),
                        52: (43, 540),
                        53: (11, 540),
                        54: (59, 560),
                        55: (27, 560),
                        56: (43, 580),
                        57: (11, 580),
                        58: (59, 600),
                        59: (27, 600),
                        60: (43, 620),
                        61: (11, 620),
                        62: (59, 640),
                        63: (27, 640),
                        64: (43, 660),
                        65: (11, 660),
                        66: (59, 680),
                        67: (27, 680),
                        68: (43, 700),
                        69: (11, 700),
                        70: (59, 720),
                        71: (27, 720),
                        72: (43, 740),
                        73: (11, 740),
                        74: (59, 760),
                        75: (27, 760),
                        76: (43, 780),
                        77: (11, 780),
                        78: (59, 800),
                        79: (27, 800),
                        80: (43, 820),
                        81: (11, 820),
                        82: (59, 840),
                        83: (27, 840),
                        84: (43, 860),
                        85: (11, 860),
                        86: (59, 880),
                        87: (27, 880),
                        88: (43, 900),
                        89: (11, 900),
                        90: (59, 920),
                        91: (27, 920),
                        92: (43, 940),
                        93: (11, 940),
                        94: (59, 960),
                        95: (27, 960),
                        96: (43, 980),
                        97: (11, 980),
                        98: (59, 1000),
                        99: (27, 1000),
                        100: (43, 1020),
                        101: (11, 1020),
                        102: (59, 1040),
                        103: (27, 1040),
                        104: (43, 1060),
                        105: (11, 1060),
                        106: (59, 1080),
                        107: (27, 1080),
                        108: (43, 1100),
                        109: (11, 1100),
                        110: (59, 1120),
                        111: (27, 1120),
                        112: (43, 1140),
                        113: (11, 1140),
                        114: (59, 1160),
                        115: (27, 1160),
                        116: (43, 1180),
                        117: (11, 1180),
                        118: (59, 1200),
                        119: (27, 1200),
                        120: (43, 1220),
                        121: (11, 1220),
                        122: (59, 1240),
                        123: (27, 1240),
                        124: (43, 1260),
                        125: (11, 1260),
                        126: (59, 1280),
                        127: (27, 1280),
                        128: (43, 1300),
                        129: (11, 1300),
                        130: (59, 1320),
                        131: (27, 1320),
                        132: (43, 1340),
                        133: (11, 1340),
                        134: (59, 1360),
                        135: (27, 1360),
                        136: (43, 1380),
                        137: (11, 1380),
                        138: (59, 1400),
                        139: (27, 1400),
                        140: (43, 1420),
                        141: (11, 1420),
                        142: (59, 1440),
                        143: (27, 1440),
                        144: (43, 1460),
                        145: (11, 1460),
                        146: (59, 1480),
                        147: (27, 1480),
                        148: (43, 1500),
                        149: (11, 1500),
                        150: (59, 1520),
                        151: (27, 1520),
                        152: (43, 1540),
                        153: (11, 1540),
                        154: (59, 1560),
                        155: (27, 1560),
                        156: (43, 1580),
                        157: (11, 1580),
                        158: (59, 1600),
                        159: (27, 1600),
                        160: (43, 1620),
                        161: (11, 1620),
                        162: (59, 1640),
                        163: (27, 1640),
                        164: (43, 1660),
                        165: (11, 1660),
                        166: (59, 1680),
                        167: (27, 1680),
                        168: (43, 1700),
                        169: (11, 1700),
                        170: (59, 1720),
                        171: (27, 1720),
                        172: (43, 1740),
                        173: (11, 1740),
                        174: (59, 1760),
                        175: (27, 1760),
                        176: (43, 1780),
                        177: (11, 1780),
                        178: (59, 1800),
                        179: (27, 1800),
                        180: (43, 1820),
                        181: (11, 1820),
                        182: (59, 1840),
                        183: (27, 1840),
                        184: (43, 1860),
                        185: (11, 1860),
                        186: (59, 1880),
                        187: (27, 1880),
                        188: (43, 1900),
                        189: (11, 1900),
                        190: (59, 1920),
                        191: (27, 1920),
                        192: (43, 1940),
                        193: (11, 1940),
                        194: (59, 1960),
                        195: (27, 1960),
                        196: (43, 1980),
                        197: (11, 1980),
                        198: (59, 2000),
                        199: (27, 2000),
                        200: (43, 2020),
                        201: (11, 2020),
                        202: (59, 2040),
                        203: (27, 2040),
                        204: (43, 2060),
                        205: (11, 2060),
                        206: (59, 2080),
                        207: (27, 2080),
                        208: (43, 2100),
                        209: (11, 2100),
                        210: (59, 2120),
                        211: (27, 2120),
                        212: (43, 2140),
                        213: (11, 2140),
                        214: (59, 2160),
                        215: (27, 2160),
                        216: (43, 2180),
                        217: (11, 2180),
                        218: (59, 2200),
                        219: (27, 2200),
                        220: (43, 2220),
                        221: (11, 2220),
                        222: (59, 2240),
                        223: (27, 2240),
                        224: (43, 2260),
                        225: (11, 2260),
                        226: (59, 2280),
                        227: (27, 2280),
                        228: (43, 2300),
                        229: (11, 2300),
                        230: (59, 2320),
                        231: (27, 2320),
                        232: (43, 2340),
                        233: (11, 2340),
                        234: (59, 2360),
                        235: (27, 2360),
                        236: (43, 2380),
                        237: (11, 2380),
                        238: (59, 2400),
                        239: (27, 2400),
                        240: (43, 2420),
                        241: (11, 2420),
                        242: (59, 2440),
                        243: (27, 2440),
                        244: (43, 2460),
                        245: (11, 2460),
                        246: (59, 2480),
                        247: (27, 2480),
                        248: (43, 2500),
                        249: (11, 2500),
                        250: (59, 2520),
                        251: (27, 2520),
                        252: (43, 2540),
                        253: (11, 2540),
                        254: (59, 2560),
                        255: (27, 2560),
                        256: (43, 2580),
                        257: (11, 2580),
                        258: (59, 2600),
                        259: (27, 2600),
                        260: (43, 2620),
                        261: (11, 2620),
                        262: (59, 2640),
                        263: (27, 2640),
                        264: (43, 2660),
                        265: (11, 2660),
                        266: (59, 2680),
                        267: (27, 2680),
                        268: (43, 2700),
                        269: (11, 2700),
                        270: (59, 2720),
                        271: (27, 2720),
                        272: (43, 2740),
                        273: (11, 2740),
                        274: (59, 2760),
                        275: (27, 2760),
                        276: (43, 2780),
                        277: (11, 2780),
                        278: (59, 2800),
                        279: (27, 2800),
                        280: (43, 2820),
                        281: (11, 2820),
                        282: (59, 2840),
                        283: (27, 2840),
                        284: (43, 2860),
                        285: (11, 2860),
                        286: (59, 2880),
                        287: (27, 2880),
                        288: (43, 2900),
                        289: (11, 2900),
                        290: (59, 2920),
                        291: (27, 2920),
                        292: (43, 2940),
                        293: (11, 2940),
                        294: (59, 2960),
                        295: (27, 2960),
                        296: (43, 2980),
                        297: (11, 2980),
                        298: (59, 3000),
                        299: (27, 3000),
                        300: (43, 3020),
                        301: (11, 3020),
                        302: (59, 3040),
                        303: (27, 3040),
                        304: (43, 3060),
                        305: (11, 3060),
                        306: (59, 3080),
                        307: (27, 3080),
                        308: (43, 3100),
                        309: (11, 3100),
                        310: (59, 3120),
                        311: (27, 3120),
                        312: (43, 3140),
                        313: (11, 3140),
                        314: (59, 3160),
                        315: (27, 3160),
                        316: (43, 3180),
                        317: (11, 3180),
                        318: (59, 3200),
                        319: (27, 3200),
                        320: (43, 3220),
                        321: (11, 3220),
                        322: (59, 3240),
                        323: (27, 3240),
                        324: (43, 3260),
                        325: (11, 3260),
                        326: (59, 3280),
                        327: (27, 3280),
                        328: (43, 3300),
                        329: (11, 3300),
                        330: (59, 3320),
                        331: (27, 3320),
                        332: (43, 3340),
                        333: (11, 3340),
                        334: (59, 3360),
                        335: (27, 3360),
                        336: (43, 3380),
                        337: (11, 3380),
                        338: (59, 3400),
                        339: (27, 3400),
                        340: (43, 3420),
                        341: (11, 3420),
                        342: (59, 3440),
                        343: (27, 3440),
                        344: (43, 3460),
                        345: (11, 3460),
                        346: (59, 3480),
                        347: (27, 3480),
                        348: (43, 3500),
                        349: (11, 3500),
                        350: (59, 3520),
                        351: (27, 3520),
                        352: (43, 3540),
                        353: (11, 3540),
                        354: (59, 3560),
                        355: (27, 3560),
                        356: (43, 3580),
                        357: (11, 3580),
                        358: (59, 3600),
                        359: (27, 3600),
                        360: (43, 3620),
                        361: (11, 3620),
                        362: (59, 3640),
                        363: (27, 3640),
                        364: (43, 3660),
                        365: (11, 3660),
                        366: (59, 3680),
                        367: (27, 3680),
                        368: (43, 3700),
                        369: (11, 3700),
                        370: (59, 3720),
                        371: (27, 3720),
                        372: (43, 3740),
                        373: (11, 3740),
                        374: (59, 3760),
                        375: (27, 3760),
                        376: (43, 3780),
                        377: (11, 3780),
                        378: (59, 3800),
                        379: (27, 3800),
                        380: (43, 3820),
                        381: (11, 3820),
                        382: (59, 3840),
                        383: (27, 3840),
                        }
            }
    return channel_groups

