# Import modules
print('Importing meta modules')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime

print('Importing Scikit-Learn modules')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

print('Importing training dataset')
# import training ms data
from binned_data_import import data
from ms_data_class import BinnedData
training_data = BinnedData(data)
X = training_data.log_transform_data
y = training_data.path
print('Data import complete')

print('Setting global variables')
from thesis_figure_parameters import catColours
today = date.today()
fig_path = './figures/'
colours = catColours
print('Global variables set')

# build model (min-maxxed)
print('Instantiating model using MinMaxScaler and LDA in Pipline')
clf = Pipeline([
    ('Min-max scaler',MinMaxScaler()),
    ('LDA', LinearDiscriminantAnalysis())
    ])
print('Fitting classifier')
print('If this is inaccurate consider building muscle mucosa only model')

clf.fit(X, y)

# Import xy data (with appropriate alignment offset value)
print("Loading Jinshi's time/position data")
time = pd.read_excel('/Users/jim/Library/CloudStorage/Box-Box/PhD/iknife-data/in-vivo-analysis/ex-vivo-model/data/griffin_tongue1_microrobot_position_data.xlsx')

# Import griffin data
print('Importing raw binned griffin data')
griffin_data_path = '/Users/jim/Library/CloudStorage/Box-Box/PhD/iknife-data/in-vivo-analysis/ex-vivo-model/data/2022_08_23_FIBRE_ROBOT_DAY_2_TONGUE_1_backgroundsubtracted.csv'
test_data = pd.read_csv(griffin_data_path)


# medlog transform and minmax griffin data
print('')

# predict per scan on griffin data an add predictions Series to xy data
# cut off based on TIC
print('')

# Plot predictions on xy data
print('')

# sense check this
print('')

# animate plot at 1hz

# overlay animated plot with main video

# Add other videos and photos if needed.
