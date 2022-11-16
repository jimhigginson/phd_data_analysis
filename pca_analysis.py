'''
Create a repeatable function using the peakpickeddata class to run a pca with scree plot, pcs and loadings to produce graphs and an output file

Means everything can be done really repeatably
'''
import seaborn as sns
from datetime import date
from thesis_figure_parameters import tfParams
import matplotlib.pyplot as plt

class PeakPickingPCAPlotter():
    '''
    This class takes a PeakPickedData object, and generates a pdf report with the unsupervised analysis, showing a scree plot, PC plots demonstrating date and class relationships, then a loadings plot with identification of the biggest factors
    '''
    # Class variable
    
    pcs_to_plot = 5
    # for the corner plots, 5 is a sweet spot of informative but not too cluttered
    fig_height = tfParams['textwidth']/pcs_to_plot
    # multifigure plots like pairplot use height ** per axis ** not overall
    today = date.today()
    sns.set_context('paper') # automatically sets axes to be best for print
    alpha = 0.5 #alpha for plots - keeps it consistent
    size = 2 #marker size in scatter plots
    marker = '.' # matplotlib marker code for scatter plots


    def __init__(self, data_object): #maybe can import rcParams here to control the graphics centrally?
        print('Initialising PCA plotting object')
        self.data = data_object
        self.pc_plot_data = self.data.principal_components.iloc[:, 0:self.pcs_to_plot].join([self.data.binary_path, self.data.path, self.data.date, self.data.energy_device])

    def __str__(self):
        return('Self encapsulated plotter function to allow easy repetition of PCA analysis')
    # Do i want to change this so it programmatically describes the data I'm plotting?

    def scree_plot(self):
        plt.clf()
        pcs_to_scree_plot = 10
        print(f'Plotting scree plot for first {pcs_to_scree_plot} principal components.')
        self.ax = sns.barplot(x = self.data.pc_labels[0:pcs_to_scree_plot], y = self.data.pca.explained_variance_ratio_[0:pcs_to_scree_plot]  * 100, color = 'k')
        self.ax.set(ylabel = 'Explained Variance (%)')
        self.scree_path = f'./figures/{self.today}_scree.pdf'
        self.fig = self.ax.get_figure()
        self.fig.figsize=(self.fig_height, self.fig_height)
        print(f'Saving scree plot to {self.scree_path}.')
        self.fig.savefig(self.scree_path)

    def binary_pc_plot(self):
        '''
        Returns a pretty corner graph of the PCs 1-5, coloured by binary tumour/non-tumour status of the tissue
        '''
        print('Plotting binary PCA plot')
        self.fig = sns.PairGrid(data = self.pc_plot_data, hue='binary_path', corner=True, height = self.fig_height, palette=['green','red'])
        self.fig.map_diag(sns.histplot)
        self.fig.map_offdiag(sns.scatterplot, markers=self.marker, alpha=self.alpha, s=self.size)
        self.fig.add_legend(title='Pathological classification')
        sns.move_legend(self.fig, 'upper center')
        self.binary_plot_path = f'./figures/{self.today}_binary_pc_plot.pdf'
        print(f'Saving binary PCA plot to {self.binary_plot_path}')
        self.fig.savefig(self.binary_plot_path)

    def date_pc_plot(self):
        '''
        Returns a pretty corner graph of the PCs 1-5, coloured by date of analysis 
        '''
        print('Plotting datewise PCA plot')
        self.fig = sns.PairGrid(data = self.pc_plot_data, hue='date', corner=True, height = self.fig_height)
        self.fig.map_diag(sns.histplot)
        self.fig.map_offdiag(sns.scatterplot, markers='.', alpha=0.5, s=2)
        self.fig.add_legend(title='Date of analysis')
        sns.move_legend(self.fig, 'upper center')
        self.date_plot_path = f'./figures/{self.today}_date_pc_plot.pdf'
        print(f'Saving date PCA plot to {self.date_plot_path}')
        self.fig.savefig(self.date_plot_path)


    def multiclass_pc_plot(self):
        '''
        Returns a pretty corner graph of the PCs 1-5, coloured by class
        '''
        print('Plotting multiclass PCA plot')
        self.fig = sns.PairGrid(data = self.pc_plot_data, hue='path', corner=True, height = self.fig_height)
        self.fig.map_diag(sns.histplot)
        self.fig.map_offdiag(sns.scatterplot, markers=self.marker, alpha=self.alpha, s=self.size)
        self.fig.add_legend(title='Pathological classification')
        sns.move_legend(self.fig, 'upper center')
        self.pathology_plot_path = f'./figures/{self.today}_pathology_pc_plot.pdf'
        print(f'Saving pathology PCA plot to {self.pathology_plot_path}')
        self.fig.savefig(self.pathology_plot_path)

    def energy_device_pc_plot(self):
        '''
        Returns a PCA plot by energy device
        '''
        print('Plotting energy device PCA plot')
        self.fig = sns.PairGrid(data = self.pc_plot_data, hue='energy_device', corner=True, height = self.fig_height)
        self.fig.map_diag(sns.histplot)
        self.fig.map_offdiag(sns.scatterplot, markers=self.marker, alpha=self.alpha, s=self.size)
        self.fig.add_legend(title='Energy device used')
        sns.move_legend(self.fig, 'upper center')
        self.energy_plot_path = f'./figures/{self.today}_energy_pc_plot.pdf'
        print(f'Saving energy device PCA plot to {self.energy_plot_path}')
        self.fig.savefig(self.energy_plot_path)



# Loadings plot
    def loadings_plot(self):
        '''
        Going to print a plot of loadings for PC1, maybe 2 and three as a horizontal strip??
        '''
        print('Holder text that will eventually verbose the plot creation')
