import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb



## create data structure to import each ion, with the cutoff to use to make the background sharp in the 'make_lupton_rgb' 'minimum' arg and a string to use as the legend handle.

features = [

        ]


## Create a function to take three features and plot them as rbg, then export a sexy diagram with nice axes, titles and legends


def ms_imaging(feature1, feature2, feature3):
    '''
    This function takes the contents of the data structure containing the file, cutoff and legend handle, and converts it into a RGB image
    '''
    ms_image = make_lupton_rgb(
            feature1,
            feature2,
            feature3,
            minimum = [
                feature1_minimum,
                feature2_minimum,
                feature3_minimum
                ]
            )
    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(gs[:-1,:]) ##for the plot
    ax2 = fig.add_subplot(gs[-1,:])   ##for the legend
    ax1.imshow(ms_image)
    

