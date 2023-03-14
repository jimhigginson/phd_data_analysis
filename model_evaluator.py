'''
in this script I'm going to create a function (or class??) that takes a pickled rfecv model and the dataset from which it was created, and generate a report of all the relevant metrics.
'''

from datetime import date
import pickle
from pp_data_import import data
from binned_data_import import data as binned_data
from ms_data_class import PeakPickedData, BinnedData
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, make_scorer, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
import matplotlib.pyplot as plt
import pandas as pd
from thesis_figure_parameters import tfParams

today = date.today()

def title(string):
    banner = 40*'#'
    print(f'\n{banner}\n'+string+f'\n{banner}\n')


# import the data and models here
data = PeakPickedData(data)
model_path = './models/'
figure_path = './figures/'

date_to_analyse = '2022-12-08'
# can adjust this if I do more runs

model_stems = [
'binary_lda_rfecv',
'binary_rf_rfecv',
'multiclass_lda_rfecv',
'multiclass_rf_rfecv'
]


# bit of code to test loading one model - delete once I get it running smoothly in a loop
binary_lda = pickle.load(open(f'{model_path}{date_to_analyse}_{model_stems[0]}.pkl','rb'))
binary_rf = pickle.load(open(f'{model_path}{date_to_analyse}_{model_stems[1]}.pkl','rb'))
multiclass_lda = pickle.load(open(f'{model_path}{date_to_analyse}_{model_stems[2]}.pkl','rb'))
multiclass_rf = pickle.load(open(f'{model_path}{date_to_analyse}_{model_stems[3]}.pkl','rb'))

binary_lda_features = pd.read_csv(f'{model_path}{date_to_analyse}_{model_stems[0]}.csv', usecols=[1], dtype='str')
binary_rf_features = pd.read_csv(f'{model_path}{date_to_analyse}_{model_stems[1]}.csv', usecols=[1], dtype='str')
multiclass_lda_features = pd.read_csv(f'{model_path}{date_to_analyse}_{model_stems[2]}.csv', usecols=[1], dtype='str')
multiclass_rf_features = pd.read_csv(f'{model_path}{date_to_analyse}_{model_stems[3]}.csv', usecols=[1], dtype='str')

binary_lda_features = binary_lda_features['0']
binary_rf_features = binary_rf_features['0']
multiclass_lda_features =  multiclass_lda_features['0']
multiclass_rf_features = multiclass_rf_features['0']

models = [
binary_lda,
binary_rf,
multiclass_lda,
multiclass_rf
]

features = [
binary_lda_features,
binary_rf_features,
multiclass_lda_features,
multiclass_rf_features
]

models[0].fit(data.data[features[0]], data.binary_path)

models[1].fit(data.data[features[1]], data.binary_path)
models[2].fit(data.data[features[2]], data.path)
models[3].fit(data.data[features[3]], data.path)

'''

######################
# Report starts here #
######################

title(f'This report was generated on {today}')
print(f'The first model is {binary_lda_file}')

title('Basic model information')

print(f'This model used the {binary_lda.estimator_} estimator to classify the signal')
print(f'It used the {binary_lda.scoring} scoring model to optimise the number of features')
print(f'The model distinguishes {len(binary_lda.classes_)} classes: {binary_lda.classes_}')

title('Recursive Feature Elimination')
print(f'The Recursive feature elimination determined that the optimal performance was achieved with {binary_lda.n_features_} features')

print(f'The features included in the model are:\n {list(binary_lda.get_feature_names_out())}')

title('Model Performance')
print(f'With these features, the cross-validated model achieved a {binary_lda.scoring} score of {round(100 * binary_lda.score(data.log_transform_data, data.binary_path), 2)}%')

print(f'Now to create a confusion matrix...')
fig = ConfusionMatrixDisplay.from_estimator(binary_lda, data.log_transform_data, data.binary_path)
plt.show()

print('And next the ROC curve')
plt.clf()
fig = RocCurveDisplay.from_estimator(binary_lda, data.log_transform_data, data.binary_path)
plt.show()
'''

class ModelEvaluator():
    '''
Class that takes the filename (without the .pkl extension) of an RFECV model, and the data and classifiers from which it was generated, and returns an evaluation of the model, with parameters, scores and graphs
    '''

    # Class variables here
    figure_width = tfParams['textwidth']
    colours = plt.cm.plasma

    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target
        # Run the functions here so it all fires up on being called

    def __str__(self):
        return(f'This class instance is evaluating the RFECV {self.model} model, fitted to predict the following {len(self.model.classes_)} classes: {self.model.classes_}')
    
    @property
    def model_performance(self):
        model_performance = round(100 * self.model.score(self.features, self.target), 2)
        return(model_performance)

    @property
    def confusion_matrix(self):
        self.cm = ConfusionMatrixDisplay.from_estimator(self.model, self.features, self.target, cmap = self.colours)
        return(self.cm)

    @property
    def roc(self):
        self.roc_plot = RocCurveDisplay.from_estimator(self.model, self.features, self.target)
        return(self.roc_plot)

    def plotter(self):
        plt.clf()
        self.fig, self.axs = plt.subplots(1, 2, figsize=(self.figure_width, self.figure_width/3))
        self.confusion_matrix.plot(ax = self.axs[0])
        self.roc.plot(ax = self.axs[1])
        return(self.fig)






#test = ModelEvaluator(models[0], data.log_transform_data[features[0]], data.binary_path)
#test2 = ModelEvaluator(models[1], data.data[features[1]], data.binary_path)
#icle = ModelEvaluator(models[2], data.log_transform_data[features[2]], data.path)
#icle2 = ModelEvaluator(models[3], data.data[features[3]], data.path)

verbosity = 0
n_jobs = 5

logocv = LeaveOneGroupOut()

scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, pos_label='Tumour', zero_division=1),
        'recall': make_scorer(recall_score, pos_label='Tumour', zero_division=1),
        'f1': make_scorer(f1_score, pos_label='Tumour', zero_division=1),
        #'roc_auc': make_scorer(roc_auc_score) 
        }

multi_av = 'weighted'
multiclass_scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average=multi_av, zero_division=1),
        'recall': make_scorer(recall_score, average=multi_av, zero_division=1),
        'f1': make_scorer(f1_score, average=multi_av, zero_division=1),
        #'roc_auc': make_scorer(roc_auc_score) 
        }



print('fitting models with rfecv features')

binary_rf.fit(data.data[features[1]], data.binary_path)
binary_lda.fit(data.data[features[0]], data.binary_path)
multiclass_rf.fit(data.data[features[3]], data.path)
multiclass_lda.fit(data.data[features[2]], data.path)

print('fitting complete')


print('Performing LOOCV and scoring binary random forest model')
b_rf_score = pd.DataFrame(cross_validate(
        estimator = binary_rf,
        X = data.data[features[1]], 
        y = data.binary_path,
        groups = data.patient_number,
        scoring = scorers,
        cv = logocv,
        verbose = verbosity,
        n_jobs = n_jobs
        ))

print('Performing LOOCV and scoring multiclass random forest model')
m_rf_score = pd.DataFrame(cross_validate(
        estimator = multiclass_rf,
        X = data.data[features[3]], 
        y = data.path,
        groups = data.patient_number,
        scoring = multiclass_scorers,
        cv = logocv,
        verbose = verbosity,
        n_jobs = n_jobs
        ))


print('Performing LOOCV and scoring binary LDA model')
b_lda_score = pd.DataFrame(cross_validate(
        estimator = binary_lda,
        X = data.data[features[0]], 
        y = data.binary_path,
        groups = data.patient_number,
        scoring = scorers,
        cv = logocv,
        verbose = verbosity,
        n_jobs = n_jobs
        ))

print('Performing LOOCV and scoring multiclass LDA model')
m_lda_score = pd.DataFrame(cross_validate(
        estimator = multiclass_lda,
        X = data.data[features[2]], 
        y = data.path,
        groups = data.patient_number,
        scoring = multiclass_scorers,
        cv = logocv,
        verbose = verbosity,
        n_jobs = n_jobs 
        ))
'''

print('Now for the binned data analysis')

binned_data = BinnedData(binned_data)
X = binned_data.data
y_1 = binned_data.binary_path
y_2 = binned_data.path

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('LDA', LinearDiscriminantAnalysis())
    ])

binned_b_lda_score = pd.DataFrame(cross_validate(
        estimator = pipe,
        X = X, 
        y = y_1,
        groups = binned_data.patient_number,
        scoring = scorers,
        cv = logocv,
        verbose = verbosity,
        n_jobs = n_jobs
        ))


binned_m_lda_score = pd.DataFrame(cross_validate(
        estimator = pipe,
        X = X, 
        y = y_2,
        groups = binned_data.patient_number,
        scoring = multiclass_scorers,
        cv = logocv,
        verbose = verbosity,
        n_jobs = n_jobs
        ))
'''

