'''
in this script I'm going to create a function (or class??) that takes a pickled rfecv model and the dataset from which it was created, and generate a report of all the relevant metrics.
'''

from datetime import date
import pickle
from pp_data_import import data
from ms_data_class import PeakPickedData
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from thesis_figure_parameters import tfParams

today = date.today()

def title(string):
    banner = 40*'#'
    print(f'\n{banner}\n'+string+f'\n{banner}\n')


# import the data and models here
data = PeakPickedData(data)
model_path = './models/'
figure_path = './figures/'
binary_lda_file = 'binary_lda_rfecv'
binary_lda = pickle.load(open(f'{model_path}{today}_{binary_lda_file}.pkl','rb'))




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
'''
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
    # perhaps text width from tfParams?
    model_path = './models/'
    figure_path = './figures/'
    today = date.today()
    figure_width = tfParams['textwidth']
    colours = plt.cm.plasma

    def __init__(self, filename, features, target):
        self.filename = filename
        self.filepath = f'{self.model_path}'+ self.filename + '.pkl'
        file = open(self.filepath, 'rb')
        self.model = pickle.load(file)
        file.close()
        self.features = features
        self.target = target
        # Run the functions here so it all fires up on being called

    @property
    def model_performance(self):
        model_performance = round(100 * self.model.score(self.features, self.target), 2)
        return(model_performance)

    def confusion_matrix(self):
        cm = ConfusionMatrixDisplay.from_estimator(self.model, self.features, self.target, cmap = self.colours)
        self.fig, self.ax = plt.subplots(figsize=(self.figure_width, self.figure_width))
        cm.plot(ax=self.ax)
        self.fig.savefig(f'{self.figure_path}{self.filename}_confusion_matrix.pdf')
