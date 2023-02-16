import pandas as pd


##########
# path in rcs directory to ex-vivo peak-picked matrix
##########

data_path = './data/combined_binned_data_metadata.csv'

print(f'Importing binned_data from {data_path}') 
data = pd.read_csv(data_path)

print(f'Binned data successfully imported\n')
