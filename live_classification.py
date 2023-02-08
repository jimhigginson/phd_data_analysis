# import relevant modules here
import pandas as pd
import pickle

# import the model and data to use

model_path = './models/2023-02-06_binary_lda_rfecv'
live_data = './data/2023-01-05_in-vivo_binned_data_medlogscale.csv'
live_metadata = './data/2023-01-05_in-vivo_binned_metadata.csv'

# unpickle the model

'''
mask_data = pd.read_csv(f'{model_path}.csv')

print(f'Unpickling model {model_path}.')

model = pickle.load(open(f'{model_path}.pkl','rb'))

print(f'Model {model_path} successfully unpickled.')

# load data
'''
print(f'Loading live data from {live_data}.')

data = pd.read_csv(live_data)

print(f'{live_data} successfully loaded.')

# load metadata

print(f'Loading live metadata from {live_metadata}.')

metadata = pd.read_csv(live_metadata)

print(f'{live_metadata} successfully loaded.')
