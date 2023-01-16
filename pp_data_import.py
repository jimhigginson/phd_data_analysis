import pandas as pd


##########
# path in rcs directory to ex-vivo peak-picked matrix
##########

data_path = './data/combined_data_metadata.csv'
#data_path = './data/combined_binned_data_metadata.csv'

print(f'Importing data from {data_path}') 
data = pd.read_csv(data_path)

print(f'Data successfully imported\n')
