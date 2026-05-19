
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
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA as sklearnPCA
# import h5py
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import fnmatch

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

def load_mat(path):
	d = {}
	f = h5py.File(path)
	ar = np.array(f['rez']['st3']).T
	tmt_arr =  np.array(f['rez']['dWU'])
	d['tmt'] = tmt_arr
	df = pd.DataFrame(ar, columns=['samples', 'spike_templates', 'a', 'b', 'cuid'])
	df['times'] = df.samples/30000.0
	
	return df, d

def getRaster_kilosort(df, cluster_number, trial_length): 
	cluster = df[df['cuid'] == cluster_number] 
	
	df = pd.DataFrame() 
	df['trial'] = (cluster.times//trial_length) 
	df['times'] =  cluster.times.sub((df.trial * trial_length), axis=0 ) 
	df.index = df.trial.values 
	 
	return df

# this functions adds zscore and ztc(zscore time course) columns to the main dataframe. Zscore used in K-Means clustering and heatmaps
#and heatmaps; Ztc to plot time series plot seaborn 

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

def unit_kmeans(data, n, ind, col,  val , key, time_idx):
	tmp = data
	df_new = tmp.pivot(index= ind, columns= col, values= val)
	
	df_new = df_new.reset_index().drop( ind,1)
	# df_new = df_new.dropna()
	df_new = df_new.T
	df_new = df_new.dropna()
	X = df_new.ix[:,time_idx].values #0.5-2 second interval
	y = df_new.index.values.tolist() # corresponding cuid
	
	
	sklearn_pca = sklearnPCA(n_components=n) #compute 3 pc
	Y_sklearn = sklearn_pca.fit_transform(X)
	pca = sklearn_pca.fit(X)
	
	print(sklearn_pca.explained_variance_ratio_) 
	
	
	#pca = PCA(n_components=n_digits).fit(data)
	model = KMeans(init=pca.components_, n_clusters=n, n_init=1)
	
	# model = KMeans(n, init='k-means++', n_init=50, max_iter=100, tol=0.00001, 
    #         precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)  # 4 clusters

	model.fit(X)
	y_kmeans = model.predict(X)
	
	unique_id = df_new.index.values
	d = map_cluster(y_kmeans, unique_id)
	tmp[str(key)] = tmp[col].map(d)
	# tmp.describe()
    #fig = plt.figure(1, figsize=(8, 6))
    #ax = Axes3D(fig, elev=-150, azim=110)

    #ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], Y_sklearn[:, 2], c=y_kmeans,   cmap='rainbow')

    #ax.set_title("First three PCA directions")
    #ax.set_xlabel("1st eigenvector")
    #ax.w_xaxis.set_ticklabels([])
    #ax.set_ylabel("2nd eigenvector")
    #ax.w_yaxis.set_ticklabels([])
    #ax.set_zlabel("3rd eigenvector")
    #ax.w_zaxis.set_ticklabels([])
	
	return tmp, d
	
# maps group assignment(found by K-Means) to CUID 
def map_cluster(y_kmeans, n):
	ls = list(y_kmeans)
	d = {}
	j = int(0)
	for i in n:
		d[i]=ls[j]
		j = int(j + 1)
	return d

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
#     y.append(dat.ix[i][dat.ix[i]>thres].index.tolist()[-1]/100.0)

# detect duration of persistent activity/oscillations after onset of tim
# returns absolute duration after stim onset in sec
# dor duration analysis units should satisfy the following: 1) 1st peak<0.7s, 
# 2) time btw peaks max = 0.5s (2Hz), 3)max duration less than 2s 

"""
from detect_peaks import detect_peaks
def _duration_peaks_pop(df):
	ls = []
	_data = df
	for j in _data.cuid.unique():
		data = _data[_data.cuid==j].zscore
	#         data = data[:250]
		thres = np.std(data[:50])

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
					continue
			elif peakind[0]>50 and peakind[0]<70:
				dur = (peakind[0]/100) - 0.5
			else:
				continue
			if dur<0 or dur>2:
				continue
		else:
			continue

		ls.append(dur) 
	return np.array(ls)
	#     plt.plot(data)
	#     plt.plot(idx, data[idx], 'ro')
	#     plt.plot(peakind, data[peakind], 'go')
	#     plt.show()

"""

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

def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir)
			if os.path.isdir(os.path.join(a_dir, name))]

def ksort_get_tmt(data, unit, templates, channel_groups):
	tmt_id = data[data.cluster_id==unit].templates.unique().tolist()
	tmt_arr = templates[tmt_id]
	tmt_arr = np.mean(tmt_arr, axis=0)
	ch_idx = tmt_arr.min(axis=0).argmin()
	depth = channel_groups['geometry'][ch_idx][1]
	tmt_avg = tmt_arr[:,ch_idx]
	return tmt_avg, depth, ch_idx
 

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


# probe 64DA (bot) (bottom) channels face experimenter, 64DB (front) face monitor
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
	return channel_groups
