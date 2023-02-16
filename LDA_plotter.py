import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from thesis_figure_parameters import catColours, tfParams

#temporarily fit my own model here
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pp_data_import import data
from ms_data_class import PeakPickedData
from datetime import date

today = date.today()
fig_path = './figures/'

data = PeakPickedData(data)
X = data.log_transform_data
y = data.path

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
X2 = clf.transform(X)

lda_data = pd.DataFrame(X2)
lda_data['Class'] = y



colours = catColours

def lda_2d_plotter(data):
    classes = data.Class.cat.categories
    data = data.groupby('Class')
    plt.figure(figsize=(tfParams['textwidth'], tfParams['textwidth']))
    axes = plt.axes()
    for x in classes:
        group = data.get_group(x)
        axes.scatter( 
                       group[0],
                       group[1],
                       label = x,
                       color = colours[x]
                       )
    axes.spines[['right', 'top']].set_visible(False)
    axes.legend()
    plt.show()
    #plt.savefig(f'{fig_path}{today}_2d_LDA.pdf')


def lda_3d_plotter(data):
    classes = data.Class.cat.categories
    data = data.groupby('Class')
    plt.figure()
    axes = plt.axes(projection='3d')
    for x in classes:
        group = data.get_group(x)
        axes.scatter3D( 
                       group[0],
                       group[1],
                       group[2], 
                       label = x,
                       color = colours[x]
                       )
    axes.legend()
    axes.view_init(elev=35, azim=-35, roll=0)
    plt.savefig(f'{fig_path}{today}_3d_LDA.pdf')

lda_2d_plotter(lda_data)

'''
plt.figure()
axes = plt.axes(projection='3d')
axes.scatter3D(fat[0], fat[1], fat[2],label='Fat', color=colours['Fat'])
axes.scatter3D(Tumour[0], Tumour[1], Tumour[2], color='Red')
axes.scatter3D(Muscle[0], Muscle[1], Muscle[2], color='Brown')
axes.scatter3D(Mucosa[0], Mucosa[1], Mucosa[2], color='Green')
axes.scatter3D(Dysplasia[0], Dysplasia[1], Dysplasia[2], color='Pink')
axes.scatter3D(Conn[0],Conn[1],Conn[2], color='Blue')
axes.legend()
plt.show()
'''
