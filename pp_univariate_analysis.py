from datetime import date
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from thesis_figure_parameters import tfParams

class PeakPickingUnivariateAnalysis():
    '''
    This class takes a PeakPickedData object, and generates a volcano plot with a list of the most important features.
    Fold change and Ttests are performed on the raw data (i.e., not median log transformed), as t-testing the log transformed data led to catastrophic cancellation and subsequent precision loss.
    p values are corrected using Benjamini-Hochberg
    '''

    # Class variables go here

    sig_threshold = -np.log10(0.05)
    fc_threshold = np.log2(2)
    today = date.today()
    text_width = tfParams['textwidth']

    def __init__(self, data_object):
        print('Starting univariate analysis')
        self.data = data_object.data
        self.binary_path = data_object.binary_path
        # think carefully about whether to do this on log transformed data or normal data (double logging??)
        self.means = self.data.join(self.binary_path).groupby('binary_path').mean()
        sns.set_context('paper')

    @property
    def fold_change(self):
        print('Calculating log2 fold change')
        fold_change = np.log2(self.means.loc['Tumour']/self.means.loc['No tumour'])
        return(fold_change)

    @property
    def t_test(self):
        print('Performing independent t-test for each feature, comparing tumour with normal')
        self.tt_data = self.data
        self.tt_data.index = self.binary_path
        t_test = ttest_ind(
                self.tt_data.loc['Tumour'],
                self.tt_data.loc['No tumour']
                )
        return(t_test)
    
    @property
    def corrected_p_values(self):
        '''
        Independent t-test values corrected using Benjamini-Hochberg false discovery rate correction
        '''
        print('Correcting for false discovery rate using Benjamini-Hochberg correction')
        corrected_p_values = -np.log10(fdrcorrection(self.t_test.pvalue)[1])
        return(corrected_p_values)

    @property
    def volcano_data(self):
        # Volcano plot goes here
        print('Collating data for volcano plot')
        volcano_data = pd.DataFrame({
            'log2fc':self.fold_change,
            'corrected_p_values':self.corrected_p_values,
            })
        volcano_data['significance'] = 'Non-significant'
        upreg_filter = (volcano_data.log2fc >= self.fc_threshold) & (volcano_data.corrected_p_values >= self.sig_threshold)
        downreg_filter = (volcano_data.log2fc <= -self.fc_threshold) & (volcano_data.corrected_p_values >= self.sig_threshold)
        volcano_data.loc[upreg_filter, ['significance']] = 'Upregulated'
        volcano_data.loc[downreg_filter, ['significance']] = 'Downregulated'
        return(volcano_data)

    def volcano_plot(self):
        alpha = 0.4
        linestyle = '--'
        colour = 'gray'
        self.fig = sns.relplot(
                data=self.volcano_data,
                x = 'log2fc',
                y = 'corrected_p_values',
                hue = 'significance',
                kind = 'scatter',
                height = self.text_width,
                aspect = 1,
                alpha = 0.7,
                s = 30, # marker size
                marker = '.'
                )
        sns.move_legend(self.fig, loc='upper left', bbox_to_anchor=(0.15, 0.9), title = None)
        self.fig.set_axis_labels('log$_2$ fold change','-log$_{10}$ corrected p value')
        plt.axhline(self.sig_threshold, alpha = alpha, ls = linestyle, color = colour)
        plt.axvline(self.fc_threshold, alpha = alpha, ls = linestyle, color = colour)
        plt.axvline(-self.fc_threshold, alpha = alpha, ls = linestyle, color = colour)
        volcano_path = f'./figures/{self.today}_volcano_plot.pdf'
        print(f'Saving to {volcano_path}')
        self.fig.savefig(volcano_path)

    def key_features():
        # take the key features from univariate analysis and print them here.
       pass
