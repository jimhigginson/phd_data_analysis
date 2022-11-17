import pp_data_import 
from ms_data_class import PeakPickedData
from pp_univariate_analysis import PeakPickingUnivariateAnalysis
from pca_analysis import PeakPickingPCAPlotter
from pp_supervised_rfecv import PeakPickModelBuilder
import pickle
from datetime import datetime, date

report_start = datetime.today()
today = date.today()

print('Starting data analysis')
print('Generating report')
data = PeakPickedData(pp_data_import.data)
pca = PeakPickingPCAPlotter(data)
uni = PeakPickingUnivariateAnalysis(data)

function_calls = [
uni.volcano_plot,
pca.scree_plot,
pca.binary_pc_plot,
pca.date_pc_plot,
pca.multiclass_pc_plot,
pca.energy_device_pc_plot,
pca.loadings_plot
]

print('Starting univariate and unsupervised analysis')
for i in function_calls:
    i()

print('Univariate and unsupervised analysis complete')

print('Starting multivariate model generation')
print('Instantiating model builder class')

modeller = PeakPickModelBuilder(data)

binary_lda = modeller.binary_lda()
#multiclass_lda = modeller.multiclass_lda()
#binary_rf = modeller.binary_rf()
#multiclass_rf = modeller.multiclass_rf()

filenames = {
    binary_lda : f'{today}_binary_lda_rfecv.pkl'
        }
model_path = './models/'

for key, value in filenames:
    filepath = f'{model_path}{value}'
    file = open(filepath, 'wb')
    pickle.dump(key, file)
    file.close()


print(f'Report generation complete at {datetime.today}.')
print(f'Report generation took {datetime.today - report_start}')
