import pandas as pd

##########
# path in rcs directory to ex-vivo peak-picked matrix
##########

data_path = 'insert data path here'

##########
# path in rcs directory to ex-vivo metadata
##########

metadata_path = 'insert metadata path here'


data = pd.read_csv(data_path)

metadata = pd.read_csv(metadata_path)


##########
# Now the data is read in, instantiate it in the data class 
# You know, the one that I'm totally about to make
##########



