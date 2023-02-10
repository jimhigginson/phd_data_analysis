import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from thesis_figure_parameters import tfParams

metadata_path  = './data/2023-01-05_in-vivo_binned_metadata.csv'

metadata = pd.read_csv(metadata_path)

grouped_burns = dict(tuple(metadata.groupby('File')))

test = grouped_burns['2021_11_25_HN010.raw']
fig_path = './figures/'
cutoff = 0.6e7

def chromatogram_plotter(file, data):
    plt.figure(figsize=(tfParams['textwidth'],3))
    plt.plot(
        'Start scan',
        'Sum.',
        data = data,
        )
    plt.fill_between(
        x = data['Start scan'],
        y1 = data['Sum.'],
        y2 = cutoff,
        where=(data['Sum.'] > cutoff),
        interpolate = True,
        data = data
        )
    plt.title(file)
    plt.savefig(f'{fig_path}chromatogram_{file}.eps')

for key, value in grouped_burns.items():
    chromatogram_plotter(key, value)
