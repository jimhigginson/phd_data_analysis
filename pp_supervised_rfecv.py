from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier

from datetime import date, datetime

class PeakPickModelBuilder():
    
    '''
    Create a class that imports a PeakPickedData object and performs supervised learning. Start with LDA and random forest and work up. Plot diagrams if necessary/possible.

    '''
    # class variables here
    features_step = 1 # reduce to 1 for final rfecv
    logocv = LeaveOneGroupOut()
    scoring = None #or accuracy or roc_auc
    n_jobs = 8 # for multi-threading the rfecv.
    verbosity = 1

    def __init__(self, data_object):
        print('Initialising supervised analysis class')
        print('Importing data object')
        self.raw_data = data_object.data
        self.data = data_object.log_transform_data
        self.path = data_object.path
        self.binary_path = data_object.binary_path
        self.patient_number = data_object.patient_number
        print(f'Building all models with recursive feature elimination with {self.features_step} features eliminated at a time, and {self.n_jobs} n_jobs for multithreading')
        print(f'For debugging purposes, {self.logocv =}')

    @property
    def binary_lda(self):
        start_time = datetime.now()
        X = self.data
        y = self.binary_path
        print(f'Instantiating LDA model and RFECV selector at {start_time} for the binary model')
        lda = LinearDiscriminantAnalysis()
        feature_selector = RFECV(
                lda,
                step = self.features_step,
                # cv = 2, # change to logocv in RCS cluster
                cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = self.n_jobs, # change to 8 in RCS cluster
                verbose = self.verbosity,
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = datetime.now()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

    @property
    def multiclass_lda(self):
        start_time = datetime.now()
        X = self.data
        y = self.path
        print(f'Instantiating LDA model and RFECV selector at {start_time} for the multiclass model')
        lda = LinearDiscriminantAnalysis()
        feature_selector = RFECV(
                lda,
                step = self.features_step,
                # cv = 2, # change to logocv in RCS cluster
                cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = self.n_jobs, # change to 8 in RCS cluster
                verbose = self.verbosity,
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = datetime.now()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

    @property
    def binary_rf(self):
        start_time = datetime.now()
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
                # cv = 2, # change to logocv in RCS cluster
                cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = self.n_jobs, # change to 8 in RCS cluster
                verbose = self.verbosity,
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = datetime.now()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)

    @property
    def multiclass_rf(self):
        start_time = datetime.now()
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
                # cv = 2, # change to logocv in RCS cluster
                cv = self.logocv.split(X, y, groups=self.patient_number),
                n_jobs = self.n_jobs, # change to 8 in RCS cluster
                verbose = self.verbosity,
                scoring = None # add in a scoring estimator
                )
        feature_selector.fit(X, y)
        end_time = datetime.now()
        print(f'Completing run at {end_time}.')
        print(f'Feature selection took {end_time - start_time} seconds.')
        return(feature_selector)
