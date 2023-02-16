# import relevant modules here
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from thesis_figure_parameters import tfParams

# import the model and data to use
model_path = './models/2023-02-06_binary_lda_rfecv'
live_data = './data/2023-01-05_in-vivo_binned_data_medlogscale.csv'
live_metadata = './data/2023-01-05_in-vivo_binned_metadata.csv'

# unpickle the model

mask_data = pd.read_csv(f'{model_path}.csv')

print(f'Unpickling model {model_path}.')

model = pickle.load(open(f'{model_path}.pkl','rb'))

print(f'Model {model_path} successfully unpickled.')

# load data

print(f'Loading live data from {live_data}.')

data = pd.read_csv(live_data)

print(f'{live_data} successfully loaded.')

# load metadata
print(f'Loading live metadata from {live_metadata}.')

metadata = pd.read_csv(live_metadata)

print(f'{live_metadata} successfully loaded.')


# Here use model.features_in_ to create a mask for live_data so the model only sees the relevant features

# doing this at this early stage should also reduce the comuptational overhead for later stages

print('Removing trailing \'000\' from data column headers to match with model features')
data.columns = data.columns.str.rstrip('000')
print('Trailing zeros successfully removed')

print('Subsetting data to only those features selected in the RFECV process of model building')
data = data[model.feature_names_in_]
print('Data subsetting complete')

print('Performing diagnostic predictions')
metadata = metadata.assign(prediction = model.predict(data))
print('Diagnostic classifications completed')
                           
# Now I need to either group by file and then apply a cutoff, or vice versa. 
# I'll do the simplest first and do a universal cutoff so at least I have the code for it laid out

cutoff = 0.6e7
'''
print(f'Filtering metadata by raw Total Ion Count, to only include those scans over the cutoff of {cutoff}')
burns = metadata[metadata['Sum.'] > cutoff]
print('Burn identification complete')

print('Matching data to burns identified from metadata')
burns_data = data[data.index.isin(burns.index)]
print('Data -> metadata matching complete')
print('Grouping metadata by filename and collating in a dict, accessible by the key of the relevant filename')
case_metadata = dict(tuple(metadata.groupby('File')))
print('Grouping and collation complete')

'''
alpha=0.2
def live_classifier_plotter(data, alpha):
    fig, ax = plt.subplots(figsize=(tfParams['textwidth'],3))
    ax.plot(
        'Start scan',
        'Sum.',
        data = data,
        linewidth=0.1
        )
    ax.fill_between(
        x = data['Start scan'],
        y1 = data['Sum.'],
        y2 = cutoff,
        where=(data['Sum.'] > cutoff),
        interpolate = True,
        data = data
        )
    ax.fill_between(
            x = data['Start scan'],
            y1 = cutoff,
            where=(data['Sum.'] > cutoff)&(data['prediction']=='No tumour'),
            interpolate=True,
            facecolor='green',
            alpha=alpha
            )
    ax.fill_between(
            x = data['Start scan'],
            y1 = cutoff,
            where=(data['Sum.'] > cutoff)&(data['prediction']=='Tumour'),
            interpolate=True,
            facecolor='red',
            alpha=alpha
            )
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Ion Count')
    fig.tight_layout(),
    fig.show()

test = case_metadata['2021_11_25_HN010.raw']
# when I come back to this, deal with first the variable errors, then find a way to do this programattically for all surgical cases, then work out how to deal with the different TIC cut off requierd for each case. SHould I choose it manually or programatically?
live_classifier_plotter(test, alpha)
