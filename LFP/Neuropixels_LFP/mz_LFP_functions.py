#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# __version_1__ = Michael Zimmerman #06.30.2021
# __version_2__ = Michael Zimmerman #11.29.2021 - updated CSD analysis function to work with Neuropixels

# This is a new .py file to house all of the functions I've created and use

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import division
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

import scipy.stats as sstat
import scipy.signal as ssig
import scipy.fftpack
from scipy.fftpack import fft, ifft
from scipy import signal

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA as sklearnPCA
# import h5py
# from mpl_toolkits.mplot3d import Axes3D
import os
import re
import fnmatch

# import Python3_icsd as icsd 
import scipy.signal as sg
import quantities as pq

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def applyCAR(data, pr=0): #this input data should be in the form (channels,samples) ie (384,7400)
    
    if pr==1:
        print('Shape before: %s' %str(data.shape))

    ch_median = np.median(data,axis=1) #subtract median across each channel
    data = data-ch_median[:,None]
    time_median = np.median(data, axis=0) #subtract median across each time sample
    data = (data.T-time_median[:,None]).T
    
    if pr==1:
        print('Shape after: %s' %str(data.shape))
    
    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def notch_filt(data, notch_freq=60, sampling_rate=2500, q_factor=30, yes_plot=0):
    """
    This function works to apply a notch filter to the input data
    
    Default parameters:
        notch_freq = 60 Hz
        sampling_rate = 2500 Hz
        q_factor = 30
        yes_plot = 0
        
    """
    #get power spectrum of the single trial trace
    time = np.arange(0,len(data))
    fourier_trans = np.fft.rfft(data)
    abs_f_t = np.abs(fourier_trans)
    p_s = np.square(abs_f_t)
    freq = np.linspace(0, sampling_rate/2, len(p_s))
    
    #apply notch filter to remove specified hz noise
    b, a = signal.iirnotch(notch_freq, q_factor, sampling_rate)
    freq_z, h = signal.freqz(b, a, fs=sampling_rate)
    data_notched = signal.filtfilt(b, a, data)
    
    #get power spectrum of the filtered trace
    fourier_trans2 = np.fft.rfft(data_notched)
    abs_f_t2 = np.abs(fourier_trans2)
    p_s2 = np.square(abs_f_t2)
    freq2 = np.linspace(0, sampling_rate/2, len(p_s2))
    
    # plotting
    if yes_plot == 1:
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[-1,0])
        ax3 = fig.add_subplot(gs[-1,1])
        ax1.plot(time, data, linewidth=1)
        ax1.plot(time, data_notched, color='green', linewidth=1)
        ax1.set_title('Unfiltered and Filtered signal')
        ax2.plot(freq, p_s)
        ax2.set_xlim([-1, 75])
        ax3.plot(freq2, p_s2, color='green')
        ax3.set_xlim([-1, 75])
    
    return data_notched

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tf_cmw(ax,df_res, num_freq = 40, method = 3, show=True):
    
    """
    performs complex Morlet wavelet convolution
    adapted fro Mike Cohen, AnalyzingNeuralTimeSeries Matlab code 
    (gives exactly the same result as Matlab, validated by Alex Pak)
    input: pd dataframe, n x m (trials x time)
    output: tf  n x m x k, method (total, phase/non-phase locked, itpc) x frequency x time

    range_cycles: cycled for Morlte wavelets, larger n of cycles -> better frequency precision, 
                                              smaller n of cycles -> beter temporal precision
    frex: range of frequencies
    """

    # global vars for wavelets
    base_idx = [0, int(400*2.5)] # this is the baseline time window [0,0.4] but in samples
    min_freq = 2
    max_freq = 80
    num_frex = num_freq
    range_cycles = [3, 10]
    down_fs = 2500 #this was changed to match the NP LFP sampling frequency

    # data info
#     Sampling_Rate = 30000
#     down_fs = Sampling_Rate/30.0
#     down_sample=False
#     if down_sample==True:
#         base_idx = [0, 400]
#         df_res = resampy.resample(df_res, Sampling_Rate, down_fs , axis=1)
#         #df_res = resampy.resample(df_res.values, Sampling_Rate, down_fs , axis=1)

    num_trials= df_res.shape[0]
    num_samples = df_res.shape[1]

    #frequencies vector
    frex = np.logspace(np.log10(min_freq),np.log10(max_freq),num_frex)
    #frex = np.linspace(min_freq,max_freq,num_frex)
    #time = np.linspace(0, num_samples, int(num_samples))
    time = np.linspace(0, num_samples/down_fs, int(num_samples) )

    #wavelet parameters
    s = np.divide(np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),num_frex), 2*np.pi*frex)
    wavtime = np.linspace(-2, 2, 4*down_fs+1)
    half_wave = (len(wavtime)-1)/2

    #FFT parameters
    nWave = len(wavtime)
    nData = num_trials * num_samples 
    # .shape[0] - trials, shape[1] - time


    # len of convolutions
    nConv = [nWave+nData-1, nWave+nData-1 ,  nWave+num_samples-1 ]
    # container for the output
    tf = np.zeros((4, len(frex),num_samples) )


    phase_loc = df_res.mean(axis=0)
    # should be equal to 0, while averaged
    non_phase_loc = df_res-phase_loc.T

    dataX = {}
    #FFT of total data
    dataX[0] = fft( np.array(df_res).flatten(), nConv[0])
    #FFT of nonphase-locked data
    dataX[1] = fft( np.array(non_phase_loc).flatten(), nConv[1])
    #FFT of ERP (phase-locked data)
    dataX[2] = fft( phase_loc ,nConv[2])
    
    tf3d = []
    
    #main loop
    for fi in range(len(frex)):

        # create wavelet and get its FFT
        # the wavelet doesn't change on each trial...
        wavelet  = np.exp(2*1j*np.pi*frex[fi]*wavtime) * np.exp(-wavtime**2/(2*s[fi]**2))

         # run convolution for each of total, induced, and evoked
        for methodi in range(method):

            # need separate FFT 
            waveletX = fft(wavelet,nConv[methodi])
            waveletX = waveletX / max(waveletX)

            # notice that the fft_EEG cell changes on each iteration
            a_sig = ifft(waveletX*dataX[int(methodi)],nConv[int(methodi)])
            a_sig = a_sig[int(half_wave): int(len(a_sig)-half_wave)]

            if methodi<2:
                a_sig = np.reshape(a_sig, (num_trials, num_samples ))
                
                # compute power
                temppow = np.mean(abs(a_sig)**2,0)
            else:
                #non-phase locked response
                temppow = abs(a_sig)**2
            if methodi==1:
                tf3d.append(abs(a_sig)**2) #TOTAL POWER(0), non-phase lock (1), phase lock (2)

            # db correct power, baseline +1 to make the same results as matlab range (1,501)
            tf[methodi,fi,:] = 10*np.log10( temppow / np.mean(temppow[base_idx[0]:base_idx[1]+1]) )
            #tf[methodi,fi,:] =  temppow / np.mean(temppow[base_idx[0]:base_idx[1]+1]) 

            # inter-trial phase consistency on total EEG
            if methodi==0:
                tf[3, fi, :] = abs(np.mean(np.exp(1j*np.angle(a_sig)),0))
    #contour_levels = np.arange(-10, 31, 1)
    contour_levels = np.arange(0, 25, 0.25) #this changes the colorbar range!!!
    if show==True:
        ax.set_yscale('log')
        ax.set_yticks(np.logspace(np.log10(min_freq),np.log10(max_freq),6))
        ax.set_yticklabels(np.round(np.logspace(np.log10(min_freq),np.log10(max_freq),6)))
        tf_plot = ax.contourf(time, frex, tf[0,:,:], 
                              contour_levels,  cmap = 'viridis' , extend = 'both') #this is where you can change the colors

        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_xticks([0,0.5,1,1.5,2,2.5])
        ax.tick_params(axis='both', which='major', labelsize=20)

        cb_tf = plt.colorbar(tf_plot, ax=ax)
        cb_tf.set_label('Power (dB)')
        cb_tf.set_ticks([0,8,16,24])

    return tf, time, frex, tf3d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_tf_values(tf, frex, time_window):
    tmp = np.dstack(tf[0,:,:])
    tmp = tmp.swapaxes(0,2)
    
    time_lower = int(time_window[0]*2500)
    time_upper = int(time_window[1]*2500)

    tmp1 = np.nanmean(tmp[(frex>=4) & (frex<8)], axis=0)[time_lower:time_upper]
    tmp2 = np.nanmean(tmp[(frex>=8) & (frex<12)], axis=0)[time_lower:time_upper]
    tmp3 = np.nanmean(tmp[(frex>=12) & (frex<30)], axis=0)[time_lower:time_upper]
    tmp4 = np.nanmean(tmp[(frex>=30) & (frex<40)], axis=0)[time_lower:time_upper]
    tmp5 = np.nanmean(tmp[(frex>=50) & (frex<70)], axis=0)[time_lower:time_upper]
    tmp6 = np.nanmean(tmp[(frex>=30) & (frex<70)], axis=0)[time_lower:time_upper]

    theta = np.nanmean(tmp1,axis=0)
    alpha = np.nanmean(tmp2,axis=0)
    beta = np.nanmean(tmp3,axis=0)
    lowgamma = np.nanmean(tmp4,axis=0)
    highgamma = np.nanmean(tmp5,axis=0)
    gamma= np.nanmean(tmp6,axis=0)

    tf_df = pd.DataFrame({#'stim':s, 
                            '4-8Hz':theta,
                            '8-12Hz':alpha,
                            '12-30Hz':beta,
                            '30-70Hz':gamma,
                            '30-40Hz':lowgamma,
                            '50-70Hz':highgamma,
                            'id':np.arange(np.shape(theta)[0])
                            })

    df = tf_df
    df1 = pd.melt(df, id_vars=['id']).sort_values(['variable','value'])

    df2 = df1[(df1['id'] == 0.0)]
    df2['value'] = df2['value'].abs()                                    # this is taking the abs() of the value found
    df2['dB']=df2['value'].apply(lambda x: 10*np.log10(x))

    return df2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Used to quantify the TF band values (uses "tf_cmw" and "get_tf_values" functions)

def TF_band_values(my_array, time_window):
    tf_q_ls = []
    for idx in range(my_array.shape[0]):
        data = my_array[idx]
        print(idx)
        reshaped_data = np.reshape(data,(1,len(data)))

        fig,ax=plt.subplots()
        tf, time, frex, tf3d = tf_cmw(ax=ax, df_res=reshaped_data, show=False)
        plt.close()
        
        foo_df = get_tf_values(tf, frex, time_window) #function defined above!

        tf_q_ls.append(foo_df)
    
    new_df = pd.concat(tf_q_ls)
    
    return new_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This is an updated CSD function that works for the Neurpropixels probe!

def df_CSD_analysis(data, axes, Channel_Number=int(384/4), colormap='jet', si=1.0/2500, up_samp=10):
    #prepare lfp data for use, by changing the units to SI and append quantities,
    #along with electrode geometry and conductivities
    lfp_data = data * 1E-6 * pq.V        # [mV] -> [V]
    z_data = np.linspace(20E-6, 3820E-6, Channel_Number) * pq.m  # [m]
    diam = 500E-6 * pq.m                              # [m]
    sigma = 0.3 * pq.S / pq.m                         # [S/m] or [1/(ohm*m)]
    sigma_top = 0. * pq.S / pq.m                      # [S/m] or [1/(ohm*m)]

    spline_input = {
      'lfp' : lfp_data,
      'coord_electrode' : z_data,
      'diam' : diam,
      'sigma' : sigma,
      'sigma_top' : sigma,
      'num_steps' : Channel_Number*up_samp+1,      # Spatial CSD upsampling to N steps
      'tol' : 1E-12,
      'f_type' : 'gaussian',
      'f_order' : (15, 5),
    }

    # Create the different CSD-method class instances. We use the class methods get_csd() and filter_csd() 
    # below to get the raw and spatially filtered versions of the current-source density estimates.
    csd_dict = dict(
      #delta_icsd = icsd.DeltaiCSD(**delta_input),
      #step_icsd = icsd.StepiCSD(**step_input),
      spline_icsd = icsd.SplineiCSD(**spline_input),
      #std_csd = icsd.StandardCSD(**std_input), 
    )


    for method, csd_obj in csd_dict.items():
        #plot LFP signal
        ax = axes[0]
        lfp_plot_data = data * 1E-3 * pq.mV #helps to keep the colorbar tidy and clean
        
        im = ax.imshow(np.array(lfp_plot_data), origin='lower', 
                       vmin=-abs(lfp_plot_data).max(), vmax=abs(lfp_plot_data).max(), 
                       cmap=colormap, interpolation='nearest')

        ax.axis(ax.axis('tight'))
        ax.set_xticks([0, 1*2500, 2*2500])
        ax.set_xticklabels([0,1,2])
        if up_samp == 2:
            ax.set_yticks([0, 50, 100, 150])
            ax.set_yticklabels([0*up_samp, 50*up_samp, 100*up_samp, 150*up_samp])
        ax.set_ylabel('Channel #')
        ax.set_xlabel('Time (s)')
        ax.set_title('LFP')
        cb = plt.colorbar(im, ax=ax, aspect=50, shrink=0.8)
        cb.set_label('LFP (%s)' % lfp_plot_data.dimensionality.string)

        
        #plot raw csd estimate
        csd = csd_obj.get_csd()
        #plot spatially filtered csd estimate
        ax = axes[1]
        csd = csd_obj.filter_csd(csd)
        
        x=np.arange(0,np.shape(csd)[1]*si,si)
        y=np.arange(0,np.shape(csd)[0]/up_samp,0.1) #downsample just for the plots
        X, Y = np.meshgrid(x,y)
        extent=(min(x),max(x),min(y),max(y)) #this has to change if origin='upper' in ax.imshow() below

        
        
        im = ax.imshow(np.array(csd), origin='lower', 
#                        vmin=-abs(csd).max(), vmax=abs(csd).max(), 
                       vmin=-5000, vmax=5000,
                       extent=extent, cmap=colormap, interpolation='nearest')

        ax.axis(ax.axis('tight'))
        ax.set_title(csd_obj.name)
        ax.set_ylabel('Channel #')
        ax.set_xlabel('Time (s)')
        if up_samp == 2:
            ax.set_yticks([0, 50, 100, 150])
            ax.set_yticklabels([0*up_samp, 50*up_samp, 100*up_samp, 150*up_samp])
        cb = plt.colorbar(im, ax=ax, aspect=50, shrink=0.8)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label('CSD (%s)' % csd.dimensionality.string)
        
        return csd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




