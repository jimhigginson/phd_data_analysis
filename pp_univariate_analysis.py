from datetime import datetime
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import numpy as np

class PeakPickingUnivariateAnalysis():
    '''
    This class takes a PeakPickedData object, and generates a volcano plot with a list of the most important features.
    Fold change and Ttests are performed on the raw data (i.e., not median log transformed), as t-testing the log transformed data led to catastrophic cancellation and subsequent precision loss.
    p values are corrected using Benjamini-Hochberg
    '''

    # Class variables go here

    sig_threshold = -np.log10(0.05)
    fc_threshold = np.log2(2)
    today = datetime.today()

    def __init__(self, data_object):
        print('Starting univariate analysis')
        self.data = data_object.data
        self.binary_path = data_object.binary_path
        # think carefully about whether to do this on log transformed data or normal data (double logging??)
        self.means = self.data.join(self.binary_path).groupby('binary_path').mean()

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
        return(volcano_data)

    def volcano_plot(self):
        pass

    def key_features():
        # take the key features from univariate analysis and print them here.
       pass
