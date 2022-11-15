'''
In this module I'm going to create (crib from my previous effort) and refine a class that can store the peak-picked data and metadata to allow easy repetition of data analysis
'''

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

class PeakPickedData(PCA):
    '''
    This class holds data and metadata to allow easy, repeatable instantiation of different datasets.

    It also performs PCA and allows rapid plotting of the results.

    '''
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self._n_PCs = 100
        self.leu_enk_dropper()
        self.pca() #runs pca method below so it's ready


    def __str__(self):
        return('Combined peak-picked matrix with metadata: data dimensions ' + str(self.data.shape))


    # Class Variables

    binary_class_integer_map = {
    'Tumour':1,
    'No tumour':-1
    }

    @property
    def log_transform_data(self):
        '''

        Log transforms the m/z matrix for better analysis.

        Taken from Yuchen's code, this function removes values where the data is zero, adds the median to all subsequent values, then logs the lot.

        Prints the shape of the output dataframe, then returns the median log transformed dataframe

        '''
        self.logOS = np.nanmedian(self.data[self.data!=0])
        # creates a median of the raw data, minus the 0 data. LogOS is what Yuchen called it, not sure what it stands for
        log_transform_data = np.log(self.data + self.logOS)
        return(log_transform_data)

    @property
    def n_PCs(self):
        '''
        Number of principal components - defaults to 10 but can be changed.
        '''
        return(self._n_PCs)

    def pca(self):
        self.pca = PCA(n_components = self.n_PCs)
        self.pc_labels = []
        for i in range(0, self.n_PCs):
            self.pc_labels.append('PC'+str(i+1))
        self.principal_components = pd.DataFrame(data = self.pca.fit_transform(self.log_transform_data.values), columns = self.pc_labels)
        self.loadings = pd.DataFrame(self.pca.components_.T, columns = self.pc_labels).set_index(self.data.columns)


    def scree_plot(self):
        self.ax = sns.barplot(x = self.pc_labels, y = self.pca.explained_variance_ratio_ * 100, color = 'k')
        return(self.ax)


    def pc_plot(self):
        '''
        Requests input from the user as to the number of PCs to plot in a pairgrid then returns a pretty corner graph
        '''
        self.pcs_to_plot = int(input('Please type how many principal components you would like to plot, based on the scree plot: '))
        self.pc_plot_data = self.principal_components.iloc[:, 0:self.pcs_to_plot].join(self.metadata.binary_path)
        self.ax = sns.pairplot(self.pc_plot_data, hue='binary_path', corner=True, markers='.', plot_kws={'alpha':0.6, 'linewidth':0}, palette=['red','green'])
        return(self.ax)


# Need to find a way to be able to re-run this with new numbers of PCs as set above, at the moment the fit can only be run once.

# next steps:
## loadings plot


