import pandas as pd


##########
# path in rcs directory to ex-vivo peak-picked matrix
##########

data_path = './data/combined_data_metadata.csv'

data = pd.read_csv(data_path)



##########
# Now the data is read in, instantiate it in the data class 
# You know, the one that I'm totally about to make
##########


print(f'Hey, here is the data_path: {data_path}.\n')
