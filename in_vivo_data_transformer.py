import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

data_filepath = 'data/2023-01-05_in-vivo_binned_data'
print('Importing data')
data = pd.read_csv(f'{data_filepath}.csv', index_col=0)
print('Data import complete')

# This is the raw binned data. 
# The plan is to transform it row-wise, to reflect an algorithm that can be applied independently to incoming data intraoperatively.

# add median, then log transform, then minmax scale

print('Median log transforming data')
data = np.log(data.add(np.nanmedian(data[data!=0], axis=1), axis=0))
print('Median log transformation complete')

print(data)

print('now trying to minmax scale it - wonder if it will work with the negative values')

data = pd.DataFrame(minmax_scale(data, axis=1), columns = data.columns)

print('Min max scale complete')

data.to_csv(f'{data_filepath}_medlogscale.csv', index=False)
print('Data saved successfully')
print(f'Data dimensions was {data.shape}')
