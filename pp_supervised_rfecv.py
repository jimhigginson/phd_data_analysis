from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut

import pickle
from time import time

class PeakPickModelBuilder():
    
    '''
    Create a class that imports a PeakPickedData object and performs supervised learning. Start with LDA and random forest and work up. Plot diagrams if necessary/possible.

    '''
    # class variables here
    features_step = 400 # reduce to 1 for final rfecv

    def __init__(self, data_object):
        print('Initialising supervised analysis class')
        print('Importing data object')
        self.raw_data = data_object.data
        self.data = data_object.log_transform_data
        self.path = data_object.path
        self.binary_path = data_object.binary_path
        self.patient_number = data_object.patient_number

    def binary_lda(self):
        start_time = time()
        X = self.data
        y = self.binary_path
        print(f'Instantiating LDA model and RFECV selector at {start_time}')
        lda = LinearDiscriminantAnalysis()
        logocv = LeaveOneGroupOut()
        feature_selector = RFECV(
                lda,
                step = self.features_step,
                cv = 2, # change to logocv in RCS cluster
                # logocv.split(features, target, groups=self.patient_number)
                n_jobs = 1, # change to 8 in RCS cluster
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = time()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

