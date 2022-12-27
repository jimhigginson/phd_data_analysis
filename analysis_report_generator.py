import pp_data_import 
from ms_data_class import PeakPickedData
from pp_univariate_analysis import PeakPickingUnivariateAnalysis
from pca_analysis import PeakPickingPCAPlotter
from pp_supervised_rfecv import PeakPickModelBuilder
import pickle
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt

report_start = datetime.today()
today = date.today()
model_path = './models/'
figure_path = './figures/'


def rfecv_plotter(rfecv, filepath):
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    print(f'Creating rfecv figure')
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(rfecv.min_features_to_select, n_scores + rfecv.min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title(f"Model performance with Recursive Feature Elimination and LOOCV.\n{rfecv.estimator}, Classes {rfecv.classes_}.") 
    print('Saving rfecv figure')
    plt.savefig(f'{figure_path}{filepath}.pdf')
    print('rfecv figure saved')

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

binary_lda = modeller.binary_lda
multiclass_lda = modeller.multiclass_lda
binary_rf = modeller.binary_rf
multiclass_rf = modeller.multiclass_rf


filenames = {
    binary_lda : f'{today}_binary_lda_rfecv',
    multiclass_lda : f'{today}_multiclass_lda_rfecv',
    binary_rf : f'{today}_binary_rf_rfecv',
    multiclass_rf : f'{today}_multiclass_rf_rfecv'
        }

for key, value in filenames.items():
    print('##########################')
    print(f'Preparing RFECV optimisation graph for {value}')
    print('##########################')
    rfecv_plotter(key, value)
    print('##########################')
    print(f'Plotting for {value} complete')
    print('##########################')
    filepath = f'{model_path}{value}.pkl'
    print('##########################')
    print(f'Opening {filepath} for pickling')
    file = open(filepath, 'wb')
    print('##########################')
    print(f'RFECV model for {key} had {key.n_features_} features')
    print(f'Exporting features for {key}')
    features = pd.DataFrame(key.get_feature_names_out())
    features.to_csv(f'{model_path}{value}.csv')
    print('Evaluating whether model was binary or multiclass for fitting with correct targets')
    if len(key.classes_) == 2:
        y = data.binary_path
    else:
        y = data.path
    print(f'Fitting {key.estimator} with selected features for pickling')
    model = key.estimator.fit(data.log_transform_data[key.get_support()], y)
    pickle.dump(model, file)
    print(f'Re-fitted {key} pickled to {filepath}')
    file.close()
    print('##########################')
    print('\n\n')




print(f'Report generation complete at {datetime.today()}.')
print(f'Report generation took {datetime.today() - report_start}')
