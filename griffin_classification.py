# Import modules
print('Importing meta modules')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
print('Instantiating model using MinMaxScaler and LDA in Pipeline')
clf = Pipeline([
    ('Min-max scaler',MinMaxScaler()),
    ('LDA', LinearDiscriminantAnalysis())
    ])
print('Fitting classifier')
print('If this is inaccurate consider building muscle mucosa only model')

clf.fit(X, y)

# Import griffin data
print('Importing raw binned griffin data')
griffin_data_path = '/Users/jim/Library/CloudStorage/Box-Box/PhD/iknife-data/in-vivo-analysis/ex-vivo-model/data/2022_08_23_FIBRE_ROBOT_DAY_2_TONGUE_1_backgroundsubtracted.csv'
g_data = pd.read_csv(griffin_data_path)
metadata_cols = ['File', 'Start scan','Sum.','retention_time']
g_metadata = g_data[metadata_cols]
g_data = g_data.drop(metadata_cols + ['Class','End scan'], axis=1)
print('Test data imported and split into metadata/data for subsequent transformation and prediction')

# medlog transform griffin data
print('Median log transforming griffin_data')
logOS = np.nanmedian(g_data[g_data!=0])
g_data = np.log(g_data + logOS)

# Import xy data (with appropriate alignment offset value)
print("Loading Jinshi's time/position data")
time = pd.read_excel('/Users/jim/Library/CloudStorage/Box-Box/PhD/iknife-data/in-vivo-analysis/ex-vivo-model/data/griffin_tongue1_microrobot_position_data.xlsx')

offset = 638.0
print(f'Offset time set at {offset} seconds')

print('Adding interpolated time points so mass spec and positional data line up')
inter_time = time.Time + 0.1
inter_time = inter_time.rename(lambda x: x + 0.5).rename('Time')
interpolate = pd.merge(time, inter_time, left_on='Time',right_on='Time', how='outer').sort_values(by='Time').reset_index(drop=True).interpolate()
interpolate['Time'] = np.around(interpolate.Time, decimals=1) + offset 
g_metadata['retention_secs'] = np.around(g_metadata.retention_time.interpolate()*60, decimals=1)
interpolate.Time = (interpolate.Time * 10).astype('int')
g_metadata.retention_secs = (g_metadata.retention_secs*10).astype('int')

g_metadata = g_metadata.set_index('retention_secs')
interpolate = interpolate.set_index('Time')

# predict per scan on griffin data an add predictions Series to xy data
# cut off based on TIC
print('Setting predictions')
g_metadata['prediction'] = clf.predict(g_data)
threshold = 2e5
print(f'TIC threshold for prediction set at {threshold:.2e}')

g_metadata = pd.merge(g_metadata, interpolate, left_index=True, right_index=True, how='inner')
g_metadata.loc[g_metadata['Sum.'] < threshold, 'prediction'] = 'No signal'


# Plot predictions on xy data
print('')

# sense check this
print('')

# animate plot at 1hz

# overlay animated plot with main video

# Add other videos and photos if needed.
