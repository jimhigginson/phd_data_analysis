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

# worked out hn010 cutoff as 6.45e6. Have set the others as this until determined otherwise
cutoffs = {
        '2021_01_07_HN001_S1.raw':6.05e6,
        '2021_01_07_HN001_S2.raw':4.00e6,
        '2021_01_21_HN002_S1.raw':4.10e6,
        '2021_01_21_HN002_S2.raw':3.20e6,
        '2021_01_21_HN003_S1.raw':2.45e6,
        '2021_02_18_HN004_S1.raw':3.10e6,
        '2021_02_23_HN005_S1.raw':6.60e6,
        '2021_02_23_HN005_S2.raw':3.75e6,
        '2021_03_18_HN006_S1.raw':2.35e6,
        '2021_03_18_HN006_S2.raw':2.45e6,
        '2021_04_15_HN007_S1.raw':1.45e7,
        '2021_10_14_HN008.raw':7.80e6,
        '2021_10_14_HN008_2.raw':6.45e6, # this is a dead file
        '2021_11_11_HN009.raw':1.10e7,
        '2021_11_25_HN010.raw':6.55e6,
        '2022_01_20_HN011.raw':6.55e6,
        '2022_01_20_HN011_2.raw':7.10e6,
        '2022_06_30_HN012.raw':5.10e6,
        '2022_06_30_HN012_2.raw':5.25e6
        }
'''
burns = metadata[metadata['Sum.'] > cutoff]

print('Matching data to burns identified from metadata')
burns_data = data[data.index.isin(burns.index)]
print('Data -> metadata matching complete')
print('Grouping metadata by filename and collating in a dict, accessible by the key of the relevant filename')
case_metadata = dict(tuple(metadata.groupby('File')))
print('Grouping and collation complete')

'''
alpha=0.6

def live_classifier_plotter(metadata, filename):
    data = metadata[metadata.File == filename]
    cutoff = cutoffs[filename]
    fig, ax = plt.subplots(figsize=(16,8))# this is what it ought to be for thesis inclusion as summary...#tfParams['textwidth'],3))
    ax.plot(
            'Start scan',
            'Sum.',
            data = data,
            color='black',
            label='_nolegend_',
            linewidth=0.2
            )
    ax.fill_between(
            x = data['Start scan'],
            y1 = data['Sum.'],
            y2 = cutoff,
            where=(data['Sum.'] > cutoff)&(data['prediction']=='No tumour'),
            interpolate=True,
            facecolor='green',
            label='Prediction: no tumour',
            alpha=alpha
            )
    ax.fill_between(
            x = data['Start scan'],
            y1 = data['Sum.'],
            y2 = cutoff,
            where=(data['Sum.'] > cutoff)&(data['prediction']=='Tumour'),
            interpolate=True,
            facecolor='red',
            label='Prediction: tumour',
            alpha=alpha
            )
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Ion Count')
    ax.axhline(y=cutoff, color='grey', alpha=0.2, label=f'Detection threshold ({cutoff:.2e})', linestyle='-')
    ax.legend()
    ax.set_title(f'Chromatogram & predictions for {filename}')
    fig.tight_layout()
    fig.show()

# when I come back to this, deal with first the variable errors, then find a way to do this programattically for all surgical cases, then work out how to deal with the different TIC cut off requierd for each case. SHould I choose it manually or programatically?





def printable(metadata, filename):
    '''
    Takes a file, applies the cutoff and generates a printable so I can go through them and manually apply the ground truth from the videos.
    '''
    data = metadata[metadata.File == filename]
    print(f'Filtering metadata by raw Total Ion Count, to only include those scans above the cutoff')
    cutoff = cutoffs[filename]
    print(f'Burn identification complete using cutoff: {cutoff:.2e}')
    data = data[data['Sum.'] > cutoff]
    data = data.drop(['File','End scan','Class','prediction'], axis=1)
    d = {'Start scan':('Start scan','first'), 'Sum.':('Sum.','sum'), 'burn length(s)':('Sum.','size')}
    print('Sum consecutive groups to only show the first scan and length of the burn') 
    data = data.groupby(data['Start scan'].diff().ne(1).cumsum()).agg(**d)
    return(data)

for i in cutoffs.keys():
     print(f'{i},\n {printable(metadata, i).to_markdown()}')
