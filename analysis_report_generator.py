import pp_data_import 
from ms_data_class import PeakPickedData
# here import univariate analysis module I'll create eventually
from pca_analysis import PeakPickingPCAPlotter

print('Starting data analysis')
print('Generating report')
data = PeakPickedData(pp_data_import.data)
pca = PeakPickingPCAPlotter(data)

function_calls = [
pca.scree_plot,
pca.binary_pc_plot,
pca.date_pc_plot,
pca.multiclass_pc_plot,
pca.energy_device_pc_plot,
pca.loadings_plot
]

for i in function_calls:
    i()

print('Report generation complete')
