import pandas as pd


##########
# path in rcs directory to ex-vivo peak-picked matrix
##########

data_path = './data/combined_binned_data_metadata.csv'

print(f'Importing binned_data from {data_path}') 
data = pd.read_csv(data_path)
print('Dropping equivocal or missing samples')

data = data[~data.path.isin(['Equivocal','No sample'])]

print('Correcting "No tumour" to "Mucosa"')

data.loc[data.path == 'No tumour', 'path'] = 'Mucosa'

print(f'Binned data successfully imported\n')
