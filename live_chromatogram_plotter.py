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
    fig, ax = plt.subplots(figsize=(tfParams['textwidth'],3))
    ax.plot(
        'Start scan',
        'Sum.',
        data = data,
        linewidth=0.1
        )
    ax.fill_between(
        x = data['Start scan'],
        y1 = data['Sum.'],
        y2 = cutoff,
        where=(data['Sum.'] > cutoff),
        interpolate = True,
        data = data
        )
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(file)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Ion Count')
    fig.tight_layout(),
    fig.savefig(f'{fig_path}chromatogram_{file}.pdf')
    fig.clf()

for key, value in grouped_burns.items():
    chromatogram_plotter(key, value)
