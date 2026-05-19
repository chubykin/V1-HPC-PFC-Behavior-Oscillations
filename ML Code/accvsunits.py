import sys
import time
import pickle
import logging
import random

import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from scipy.stats import kstest
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import balanced_accuracy_score

from LemonPy.utils_vmk import set_pub_plots

class AvsUplot():

    def __init__(self, region='v1', min_units_count=75):
        self.random_state = 42   
        self.stim_order = [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]

        # self.win = (0.5, 1.5) # Window of activity considered
        self.n_times_sampling = 2000 # Number of times we do random sampling; R2000
        self.region = region # Region of interest
        self.min_units_count = min_units_count
        self.units_count_array = {'hippo': [2, 4, 8, 16, 32, 50], 'v1': [2, 4, 8, 16, 32, 64, 75]}
        self.max_iter = 3000
        self.num_trials = 150

    def get_scores(self, tot_data):
        
        y = np.array(self.stim_order)
        reg_params = {'WT': [], 'FX': []}
        
        # For n=1 classify using each unit
        acc_test = {'WT': [], 'FX': []}

        for strain in ['WT', 'FX']:
            tot_muid = [i for i in tot_data if i[0] == strain]

            logging.info(strain)
            for n_units in self.units_count_array[self.region]:

                average_test_score = [] # Stores scores for each random sampling

                for sampling_iter in range(self.n_times_sampling):
                    _, _, df = random.sample(tot_muid, k=1)[0]
                    m_uid = df.cuid.unique()
                    # units_roi = df[(df.times >= self.win[0]) & (df.times < self.win[1])]
                    r_uid = np.random.choice(m_uid, size=n_units, replace=False)
                    sampled_data = df[df.cuid.isin(r_uid)]
                    sampled_data = sampled_data.pivot_table(index=['trial', 'cuid'], columns='times', 
                                                            values='spike_count', observed=False)
                    
                    # sampled_data = sampled_data.sum(axis=1) # No-Bins

                    X = sampled_data.values.reshape(self.num_trials, -1, 20, 5).sum(axis=3).reshape(self.num_trials, -1)
                    
                    model = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', 
                                                    scoring='balanced_accuracy', solver='lbfgs', 
                                                 penalty='l2', max_iter=self.max_iter)

                    scores = cross_validate(model, X, y, cv=KFold(n_splits=5, shuffle=True), 
                                                                scoring='balanced_accuracy', return_estimator=True)
                    reg_params[strain].append([scores['estimator'][i].C_ for i in range(5)])
                    
                    average_test_score.append(scores['test_score'] * 100)
                    if sampling_iter%100 == 0:
                        logging.info(sampling_iter)
                acc_test[strain].append(np.stack(average_test_score))
                logging.info(n_units)

        return acc_test, reg_params


    def save_figs(self, acc_dict, save_filename=None):
        plt.figure(figsize=(9, 7))
        data = {}
        data['FX'] = np.array(acc_dict['FX']).mean(axis=2)
        data['WT'] = np.array(acc_dict['WT']).mean(axis=2)

        logging.info(f"FX scores shape: {data['FX'].shape} WT scores shape: {data['WT'].shape}")

        for mouse_type in ['WT', 'FX']:
            mean = data[mouse_type].mean(axis=1)
            std = data[mouse_type].std(axis=1)
            sem = std / np.sqrt(data[mouse_type].shape[1])
            
            logging.info(f'{mouse_type} Max: {mean.max()}')
            plt.errorbar(self.units_count_array[self.region], mean, marker='.', yerr=sem, label=mouse_type, elinewidth=1)
        
        x_upper_lim = self.units_count_array[self.region][-1]#round(len(mean) / 10) * 10
        
        plt.yticks(np.arange(60, 100 + 1, 10))
        plt.xlim([0, x_upper_lim + 5])
        plt.xticks(np.arange(0, x_upper_lim + 10, 25))
        plt.xlabel('Number of units')
        plt.ylabel('Accuracy (%)')
        plt.title('Units vs accuracy')
        plt.legend(loc='lower right', fontsize=28, frameon=False, handlelength=1)
        sns.despine()
        plt.tight_layout()

        if save_filename is not None:    
            plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    
        for i, j in enumerate(self.units_count_array[self.region]):
            logging.info(f"{stats.shapiro(data['FX'][i, :]).pvalue > 0.05} {stats.shapiro(data['WT'][i, :]).pvalue > 0.05}")
            logging.info(f"{j} {stats.mannwhitneyu(data['FX'][i, :], data['WT'][i, :], alternative='two-sided')}\n")

if __name__ == '__main__':
    """
    Run as:

        python accvsunits.py load_filename.parquet save_filename region

    load_filename.parquet: spike count dataframe in parquet format
    save_filename: detailed filename
    region: brain region (v1, hippo)


    Example:

        python accvsunits.py operant_spikecounts.parquet v1_Binned50ms_R2000CV10 v1

    """

    start = time.time()

    pal = [(0.00784313725490196, 0.9921568627450981, 1.0), (0.9941407151095732, 0.0, 0.9941407151095732)]
    set_pub_plots(pal)

    load_filename = sys.argv[1]
    save_filename = sys.argv[2]
    region = sys.argv[3]
    min_units_count = {'hippo': 50, 'v1': 75}

    save_filename = 'Figures_run/' + save_filename
    decoder = AvsUplot(region, min_units_count[region])

    logging.basicConfig(
        filename=save_filename + '.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info(f'load_filename: {load_filename} save_filename: {save_filename} region: {region} min_units_count: {min_units_count[region]}')

    spike_counts_df = pd.read_parquet(load_filename)
    spike_counts_df = spike_counts_df[(spike_counts_df.region == region) & (spike_counts_df.visRes == 'yes')]

    sub_cuid = spike_counts_df.groupby('et').cuid.nunique() >= min_units_count[region]
    spike_counts_df = spike_counts_df[spike_counts_df.et.isin(sub_cuid[sub_cuid == True].index)]
    spike_counts_df = spike_counts_df[(spike_counts_df.times >= 0.5) & (spike_counts_df.times < 1.5)]

    logging.info(spike_counts_df.groupby(['strain', 'et']).cuid.nunique())

    tot_data = [(strain, et, datum) for (strain, et), datum in spike_counts_df.groupby(['strain', 'et'])]
    
    acc_dict = decoder.get_scores(tot_data)

    logging.info(f"Time taken: {time.time() - start}")
    
    with open(save_filename + '.pickle', 'wb') as f:
        pickle.dump(acc_dict, f)
     
    # with open(save_filename + '.pickle', 'rb') as f:
    #     acc_dict = pickle.load(f)
    
    acc_dict = acc_dict[0]

    decoder.save_figs(acc_dict, save_filename + '.pdf') 
    