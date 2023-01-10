import pandas as pd


##########
# path in rcs directory to ex-vivo peak-picked matrix
##########

data_path = './data/combined_data_metadata.csv'
binned_data_path = './data/combined_binned_data_metadata.csv'

print(f'Importing peak-picked data from {data_path} and binned data from {binned_data_path}')

data = pd.read_csv(data_path)
binned_data = pd.read_csv(binned_data_path)

print(f'Data successfully imported\n')
