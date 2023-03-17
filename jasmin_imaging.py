import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.visualization import make_lupton_rgb



## create data structure to import each ion, with the cutoff to use to make the background sharp in the 'make_lupton_rgb' 'minimum' arg and a string to use as the legend handle.

im_path = './data/imaging_data/'


features_data = {
    'Adenine':400,
    'Cer(d34_1)':100,
    'Cer(d42_1)':100,
    'FA(16_0)':100,
    'FA(18_1)':100,
    'FA(18_2)':100,
    'Glucose':100,
    'Glutamate':100,
    'Glutamine':100,
    'Glutathione':100,
    'Histidine':100,
    'Malic_acid':300,
    'PE(36_1)':100,
    'PE(36_2)':100,
    'PE(38_4)':100,
    'PE(O-36_5)':100,
    'PE(O-38_5)':100,
    'PI(38_4)':100,
    'Pantothenic_acid':100,
    'Taurine':400,
        }
#features = pd.DataFrame(features_data, columns=['feature','threshold'])

## Create a function to take three features and plot them as rbg, then export a sexy diagram with nice axes, titles and legends


def ms_imaging(feature1, feature2, feature3):
    '''
    This function takes the contents of the data structure containing the file, cutoff and legend handle, and converts it into a RGB image
    '''
    feature1_data = pd.read_csv(f'{im_path}{feature1}.csv', header=None)
    feature2_data = pd.read_csv(f'{im_path}{feature2}.csv', header=None)
    feature3_data = pd.read_csv(f'{im_path}{feature3}.csv', header=None)
    ms_image = make_lupton_rgb(
            feature1_data,
            feature2_data,
            feature3_data,
            Q = 100,
            minimum = [
                features_data[feature1],
                features_data[feature2],
                features_data[feature3]
                ]
            )
    legend_elements = [
        Patch(facecolor='red',label=f'{feature1}'),
        Patch(facecolor='green',label=f'{feature2}'),
        Patch(facecolor='blue',label=f'{feature3}'),

            ]
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot()
    ax.imshow(ms_image)
    ax.legend(handles=legend_elements)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False, # labels along the bottom edge are off
    labelleft=False)
    plt.show()
    

