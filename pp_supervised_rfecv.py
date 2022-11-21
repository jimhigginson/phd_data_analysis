from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier

from time import time

class PeakPickModelBuilder():
    
    '''
    Create a class that imports a PeakPickedData object and performs supervised learning. Start with LDA and random forest and work up. Plot diagrams if necessary/possible.

    '''
    # class variables here
    features_step = 200 # reduce to 1 for final rfecv
    logocv = LeaveOneGroupOut()
    scoring = 'balanced_accuracy' #or accuracy or roc_auc

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
        print(f'Instantiating LDA model and RFECV selector at {start_time} for the binary model')
        lda = LinearDiscriminantAnalysis()
        feature_selector = RFECV(
                lda,
                step = self.features_step,
                cv = 2, # change to logocv in RCS cluster
                # cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = 2, # change to 8 in RCS cluster
                scoring = self.scoring # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = time()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

    def multiclass_lda(self):
        start_time = time()
        X = self.data
        y = self.path
        print(f'Instantiating LDA model and RFECV selector at {start_time} for the multiclass model')
        lda = LinearDiscriminantAnalysis()
        feature_selector = RFECV(
                lda,
                step = self.features_step,
                cv = 2, # change to logocv in RCS cluster
                # cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = 2, # change to 8 in RCS cluster
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = time()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

    def binary_rf(self):
        start_time = time()
        X = self.data
        y = self.binary_path
        print(f'Instantiating random forest model and RFECV selector at {start_time} for the binary model')
        rf = RandomForestClassifier(
                #verbose=1,
                n_jobs=1
                )
        feature_selector = RFECV(
                rf,
                step = self.features_step,
                cv = 2, # change to logocv in RCS cluster
                # cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = 2, # change to 8 in RCS cluster
                # verbose = 1,
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = time()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

    def multiclass_rf(self):
        start_time = time()
        X = self.data
        y = self.path
        print(f'Instantiating Random Forest model and RFECV selector at {start_time} for the multiclass model')
        rf = RandomForestClassifier(
                # verbose=1,
                n_jobs=1
                )
        feature_selector = RFECV(
                rf,
                step = self.features_step,
                cv = 2, # change to logocv in RCS cluster
                # cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = 2, # change to 8 in RCS cluster
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = time()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)


