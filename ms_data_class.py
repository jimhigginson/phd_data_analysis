import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

class PeakPickedData(PCA):
    '''
    This class holds data and metadata to allow easy, repeatable instantiation of different datasets.

    It also performs PCA and allows rapid plotting of the results.

    On instantiating the class, it expects a pandas object with data with matching metadata including patient number, sample number, energy device, date, path, binary path, filename and presumed class
    '''

    def __init__(self, raw_data):
        self.raw_data = raw_data
        self._n_PCs = 100
        self._data = self.data
        self.pca() #runs pca method below so it's ready


    def __str__(self):
        return('Combined peak-picked matrix with metadata: data dimensions ' + str(self.data.shape))


    # Class Variables

    binary_class_integer_map = {
    'Tumour':1,
    'No tumour':-1
    }

    metadata_columns = [
        'patient_number',
        'sample_number',
        'energy_device',
        'path',
        'binary_path',
        'filename',
        'presumed_class',
        'date'
            ]

    @property
    def patient_number(self):
        '''
        Pulls the patient numbers as a series for grouping and cross-validation models
        '''

        patient_number = self.raw_data['patient_number'].astype('category')
        return(patient_number)

    @property
    def path(self):
        '''
        Pulls the pathological diagnosis as a pandas series
        '''
        path = self.raw_data['path'].astype('category')
        return(path)

    @property
    def binary_path(self):
        '''
        Pulls the tumour/non-tumour binary pathology as a series
        '''
        binary_path = self.raw_data['binary_path'].astype('category')
        return(binary_path)

    @property
    def date(self):
        '''
        Pulls the date on which the sample was run
        '''
        date = self.raw_data['date']
        return(date)

    @property
    def data(self):
        '''

        Removes the metadata columns from the raw input object to leave peak picked data
        Runs at initialisation and does not require re-running

        '''

        data = self.raw_data.drop(self.metadata_columns, axis=1)
        return(data)

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
        Number of principal components - defaults to 100 but can be changed.
        '''
        return(self._n_PCs)

    def pca(self):
        self.pca = PCA(n_components = self.n_PCs)
        self.pc_labels = []
        for i in range(0, self.n_PCs):
            self.pc_labels.append('PC'+str(i+1))
        self.principal_components = pd.DataFrame(data = self.pca.fit_transform(self.log_transform_data.values), columns = self.pc_labels)
        self.loadings = pd.DataFrame(self.pca.components_.T, columns = self.pc_labels).set_index(self.data.columns)
