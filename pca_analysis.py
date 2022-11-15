'''
Create a repeatable function using the peakpickeddata class to run a pca with scree plot, pcs and loadings to produce graphs and an output file

Means everything can be done really repeatably
'''
import seaborn as sns
from datetime import date
from thesis_figure_parameters import tfParams

class PeakPickingPCAPlotter():
    '''
    This class takes a PeakPickedData object, and generates a pdf report with the unsupervised analysis, showing a scree plot, PC plots demonstrating date and class relationships, then a loadings plot with identification of the biggest factors
    '''
    # Class variable
    
    pcs_to_plot = 5
    fig_height = tfParams['textwidth']/pcs_to_plot
    # multifigure plots like pairplot use height ** per axis ** not overall
    today = date.today()
    sns.set_context('paper')
    # for the corner plots, 5 is a sweet spot of informative but not too cluttered

    def __init__(self, data_object): #maybe can import rcParams here to control the graphics centrally?
        print('Initialising PCA plotting object')
        self.data = data_object
        self.pc_plot_data = self.data.principal_components.iloc[:, 0:self.pcs_to_plot].join([self.data.binary_path, self.data.path, self.data.date])

    def __str__(self):
        return('Self encapsulated plotter function to allow easy repetition of PCA analysis')
    # Do i want to change this so it programmatically describes the data I'm plotting?

    def scree_plot(self):
        print('Plotting scree plot')
        self.ax = sns.barplot(x = self.data.pc_labels, y = self.data.pca.explained_variance_ratio_ * 100, color = 'k')
        return(self.ax)


    def binary_pc_plot(self):
        '''
        Returns a pretty corner graph of the PCs 1-5, coloured by binary tumour/non-tumour status of the tissue
        '''
        print('Plotting binary PCA plot')
        self.fig = sns.pairplot(data = self.pc_plot_data, hue='binary_path', corner=True, markers='.', height = self.fig_height, plot_kws={'alpha':0.6, 'linewidth':0, 'legend':False}, palette=['red','green'])
        sns.move_legend(self.fig, 'top right')
        self.fig.add_legend(title='Pathological classification')
        #self.fig.set_title('Title here')
        self.binary_plot_path = f'./figures/{self.today}_binary_pc_plot.pdf'
        print(f'Saving binary PCA plot to {self.binary_plot_path}')
        #self.fig.savefig(self.binary_plot_path)
        return(self.fig)

    def date_pc_plot(self):
        '''
        Returns a pretty corner graph of the PCs 1-5, coloured by date of analysis 
        '''
        print('Plotting datewise PCA plot')
        self.ax = sns.pairplot(data = self.pc_plot_data, hue='date', corner=True, markers='.', plot_kws={'alpha':0.6, 'linewidth':0})
        return(self.ax)

    def multiclass_pc_plot(self):
        '''
        Returns a pretty corner graph of the PCs 1-5, coloured by class
        '''
        print('Plotting multiclass PCA plot')
        self.ax = sns.pairplot(data = self.pc_plot_data, hue='path', corner=True, markers='.', plot_kws={'alpha':0.6, 'linewidth':0})
        return(self.ax)

# Loadings plot
    def loadings_plot(self):
        '''
        Going to print a plot of loadings for PC1, maybe 2 and three as a horizontal strip??
        '''
        print('Holder text that will eventually verbose the plot creation')
