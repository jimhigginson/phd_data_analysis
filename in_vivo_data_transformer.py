import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

print('Importing data')
data = pd.read_csv('data/2023-01-05_in-vivo_binned_data.csv', index_col=0)
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

data.to_csv('data/2023-01-05_in_vivo_binned_medlogscale.csv', index=False)
