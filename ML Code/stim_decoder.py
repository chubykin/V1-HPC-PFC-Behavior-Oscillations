# import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, LogisticRegressionCV
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import balanced_accuracy_score
import pickle
import sys
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns
from LemonPy.utils_vmk import set_pub_plots, generate_2_pair_combinations
import time
from scipy.stats import mannwhitneyu
import logging

class StimClassifier():
    def __init__(self, region):
        self.step = 0.01
        self.wins = [(0, 0.5), (0.5, 0.7), (0.7, 1.5), (1.5, 3)]
        self.plots_data_stim_cf = pd.DataFrame()
        self.mice_count = {'WT': 10, 'FX': 8}
        self.stim_order = [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        self.num_trials = 150
        self.region = region

    def run_FWdecoder(self, spike_counts_df):
        # Fixed windows
        max_iter = 3000
        
        cm = {}
        acc_test = {}
        for win in self.wins:
            # cm[win] = {}
            acc_test[win] = {}
            win_len = round((win[1]*1000 - win[0] * 1000) / 50) # num of 10ms windows in ROI
            for key, df in spike_counts_df.groupby('strain'):
                acc_test[win][key] = {}
                start = time.time()
                for mice_et, mouse_df in df.groupby('et'):
    
                    acc_lr = {'Train': [], 'Test': []}
    
                    data = mouse_df[(mouse_df.times >= win[0]) & (mouse_df.times < win[1])]
                    data = data.pivot_table(index=['trial', 'cuid'], columns='times', values='spike_count')
                    
                    X = data.values.reshape(self.num_trials, -1, win_len, 5).sum(axis=3).reshape(self.num_trials, -1)

                    y = np.array(self.stim_order)
                    
                    model = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', 
                                                    scoring='balanced_accuracy', solver='lbfgs', penalty='l2')

                    scores = cross_validate(model, X, y, cv=KFold(n_splits=5, shuffle=True), 
                                                            scoring='balanced_accuracy', return_estimator=True)

                    acc_test[win][key][mice_et] = scores['test_score'] * 100 
                    
                logging.info(f'Done Training..{key}_{win}  Time taken: {time.time() - start:.2f}s')
        logging.info('ALL DONE!')
        return acc_test
    
    def run_SWdecoder(self, spike_counts_df):
        # Fixed windows
        win = 0.05
        max_iter = 5000
        acc_test = {}
        times = np.round(np.arange(0., 3 - win, self.step), 2)  
        for key, df in spike_counts_df.groupby('strain'):
            acc_test[key] = {}
            start = time.time()
            for mice_et, mouse_df in df.groupby('et'):
                acc_test[key][mice_et] = []
                for roi in times:    
                    data = mouse_df[(mouse_df.times >= roi) & (mouse_df.times < round(roi + win, 2))]
                    data = data.pivot_table(index=['trial', 'cuid'], columns='times', values='spike_count')
                    
                    X = data.values.sum(axis=1).reshape(self.num_trials, -1)
                    y = np.array(self.stim_order)
                    
                    model = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=max_iter,
                                                scoring='balanced_accuracy', solver='lbfgs', penalty='l2')
                    
                    scores = cross_validate(model, X, y, cv=KFold(n_splits=5, shuffle=True), 
                                            scoring='balanced_accuracy')
                    
                    acc_test[key][mice_et].append(scores['test_score'] * 100) 

                logging.info(f'Done Training..{key}_{mice_et}  Time taken: {time.time() - start:.2f}s')
        logging.info('ALL DONE!')
        return acc_test

    def save_fig_FW(self, acc_test, save_filename):
    
        plt.figure(figsize= (12, 5))
        x_axis_labels = self.wins
    
        y_WT = np.array([np.mean(np.array(list(acc_test[win]['WT'].values())).mean(axis=1)) for win in self.wins])
        y_FX = np.array([np.mean(np.array(list(acc_test[win]['FX'].values())).mean(axis=1)) for win in self.wins])
    
        yerr_WT = np.array([np.std(np.array(list(acc_test[win]['WT'].values())).mean(axis=1)) for win in self.wins]) / np.sqrt(10)
        yerr_FX = np.array([np.std(np.array(list(acc_test[win]['FX'].values())).mean(axis=1)) for win in self.wins]) / np.sqrt(8)
    
    
        logging.info(f'WT: {y_WT.max():.2f}%, FX: {y_FX.max():.2f}%')
    
        x_axis = np.arange(len(self.wins))
    
        plt.errorbar(x_axis, y_WT, marker='.', yerr=yerr_WT, capsize=5, elinewidth=1)
        plt.errorbar(x_axis + 0.05, y_FX, marker='.', yerr=yerr_FX, capsize=5, elinewidth=1)
        plt.xticks(x_axis, x_axis_labels)
        plt.xlabel('Time windows')
        plt.ylabel('Accuracy (%)')
        sns.despine()
        plt.legend(['WT', 'FX'], loc='upper right', fontsize=26, frameon=False, handlelength=1)
        plt.title('Stimuli decoding: Fixed windows')
        plt.yticks(np.arange(40, 110, 20))
        logging.info(f"WT: {y_WT} \nFX: {y_FX}")
        plt.savefig(save_filename + '.pdf', bbox_inches='tight', dpi=300)
        
        for win in self.wins:
           logging.info(f"Window: {win} {mannwhitneyu(np.array(list(acc_test[win]['WT'].values())).mean(axis=1), np.array(list(acc_test[win]['FX'].values())).mean(axis=1))}")

        for strain, (y, yerr, color) in {'WT': [y_WT, yerr_WT, 'cyan'], 'FX': [y_FX, yerr_FX, 'magenta']}.items():
            plt.figure(figsize=(8, 7))
            plt.bar([1, 2, 3, 4], y, yerr=yerr, capsize=5, width=0.75, color=color)
            sns.despine()
            plt.xticks([1, 2, 3, 4], self.wins, rotation=45)
            plt.yticks(np.arange(0, 100.1, 20))
            plt.xlabel('Strain')
            plt.ylabel('Accuracy (%)')
            plt.savefig(save_filename + f'_{self.region}_{strain}_stimuli_quantification.pdf', dpi=300, bbox_inches='tight')
        
        for strain in ['FX', 'WT']:
            for win1, win2 in generate_2_pair_combinations(self.wins):
                logging.info(f"{strain} Windows: {win1}, {win2}, {mannwhitneyu(np.array(list(acc_test[win1][strain].values())).mean(axis=1), np.array(list(acc_test[win2][strain].values())).mean(axis=1))}")
 
    def save_fig_SW(self, acc_test, save_filename):
        win = 0.05
        times = np.arange(0., 3 - win, self.step)
        plots_data_stim_cf = pd.DataFrame(index=times)
        for key in ['FX', 'WT']:
            plots_data_stim_cf[f'Overall_mean_{key}_0.05s'] = np.stack(list(acc_test[key].values())).mean(axis=(0, 2))
            plots_data_stim_cf[f'Overall_std_{key}_0.05s'] = np.stack(list(acc_test[key].values())).mean(axis=2).std(axis=0)
        
        plots_data_stim_cf.to_csv(save_filename + '.csv', index=True, index_label='time')
        
       # Plot all accuracy graphs for each window
        plt.figure(figsize= (20, 6))
        std_factor = {'WT': 10, 'FX': 8}
        for key in ['WT', 'FX']:
            mean = plots_data_stim_cf[f'Overall_mean_{key}_{win}s'].values
            std = plots_data_stim_cf[f'Overall_std_{key}_{win}s'].values
            sns.lineplot(plots_data_stim_cf, x=plots_data_stim_cf.index, y=f"Overall_mean_{key}_{win}s", label=key)
            plt.fill_between(plots_data_stim_cf.index, mean - std/np.sqrt(std_factor[key]), mean + std/np.sqrt(std_factor[key]), alpha=.2)

        plt.title(f'Stimuli decoding: SW decoder')
        plt.fill_between([0.5, 0.7], [100, 100], color='gray', alpha = 0.4)
        plt.ylim([40, 100])
        plt.legend(loc='upper right', fontsize=28, frameon=False, handlelength=1)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Time (s)')
        sns.despine()
        plt.savefig(save_filename + '.pdf', dpi=300)

if __name__ == '__main__':
    """
    Command line:
        python stim_decoder.py load_filename.parquet region save_filename type
    
    load_filename: spike count dataframe
    save_filename: detailed name of file
    region: v1 or hippo
    type: SW (sliding window) or FW (fixed-window)
    
    Example:
        >> python stim_decoder.py operant_spikecounts.parquet v1 v1_FWD_Binned50ms FW
        >> python stim_decoder.py operant_spikecounts.parquet hippo v1_SWD_Binned50ms SW
    """
    start = time.time()

    pal = [(0.00784313725490196, 0.9921568627450981, 1.0), (0.9941407151095732, 0.0, 0.9941407151095732)]
    set_pub_plots(pal)

    load_filename = sys.argv[1]
    region = sys.argv[2]
    save_filename = sys.argv[3]
    decoder_type = sys.argv[4]
    save_filename = 'Figures/' + save_filename

    logging.basicConfig(
        filename=save_filename + '.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info(f'Load filename: {load_filename}, Save filename: {save_filename} Region: {region}')

    spike_counts_df = pd.read_parquet(load_filename)
    spike_counts_df = spike_counts_df[(spike_counts_df.region == region) & (spike_counts_df.visRes == 'yes')]
    logging.info(f"WT units: {spike_counts_df[spike_counts_df.strain == 'WT'].cuid.nunique()} \
                   FX units: {spike_counts_df[spike_counts_df.strain == 'FX'].cuid.nunique()}")

    decoder = StimClassifier(region)
    
    if decoder_type == 'FW':
        acc_test = decoder.run_FWdecoder(spike_counts_df)
        
        with open(save_filename  + '.pickle', 'wb') as f:
            pickle.dump(acc_test, f)

        ## If you already have the pickle file saved and just want to create figures
        # with open(save_filename  + '.pickle', 'rb') as f:
        #     acc_test = pickle.load(f)
            
        decoder.save_fig_FW(acc_test, save_filename)

    elif decoder_type == 'SW':
        
        acc_test = decoder.run_SWdecoder(spike_counts_df)
        
        with open(save_filename  + '.pickle', 'wb') as f:
            pickle.dump(acc_test, f)

        ## If you already have the pickle file saved and just want to create figures uncomment
        # with open(save_filename  + '.pickle', 'rb') as f:
        #     acc_test = pickle.load(f)

        decoder.save_fig_SW(acc_test, save_filename)

    logging.info('Total time taken (s): {}'.format(time.time() - start))

    