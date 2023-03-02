print('Importing meta modules')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import date, datetime

print('Importing Scikit-Learn modules')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, train_test_split, cross_val_score, StratifiedKFold, LearningCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, auc

print('Importing my own scripts and data modules')
from thesis_figure_parameters import catColours, tfParams
from pp_data_import import data
from ms_data_class import PeakPickedData
print('Modules imported successfully')

print('Setting global variables')
today = date.today()
fig_path = './figures/'
colours = catColours
print('Global variables set')

print('Setting custom functions')

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
    plt.savefig(f'{fig_path}{today}_2d_LDA.pdf')


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


def bin_lda_plotter(data):
    classes = data.Class.cat.categories
    data = data.groupby('Class')
    plt.figure(figsize=(tfParams['textwidth'], 3))
    ax = plt.axes()
    for x in classes:
        group = data.get_group(x)
        ax.scatter(
            group[0],
            group[0],
            label = x,
            color = colours[x]
            )
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)
    plt.savefig(f'{fig_path}{today}_binary_lda_plot.pdf')

print('Custom functions created')

start = datetime.now()

print(f'Importing and organising data at {start}')
data = PeakPickedData(data)

X = data.log_transform_data
y = data.binary_path

logocv = LeaveOneGroupOut()
cv = StratifiedKFold(n_splits=10)
groups = data.raw_data.patient_number

print(f'Data import and organisation complete at {datetime.now()}, that took {datetime.now() - start}')

print('Instantiating LDA model')
clf = LinearDiscriminantAnalysis()
print('Fitting model to binary target')
clf.fit(X, y)


'''
X2 = clf.transform(X)
bin_lda_data = pd.DataFrame(X2)
bin_lda_data['Class'] = y

start=datetime.now()
print(f'Starting generation of LOOCV learning curve at {start}')

fig, ax = plt.subplots(figsize=(tfParams['textwidth'],tfParams['textwidth']))
LearningCurveDisplay.from_estimator(
        clf,
        X,
        y,
        cv=logocv,
        groups=groups,
        score_type='both',
        train_sizes=np.linspace(0.03, 1.0, 40),
        n_jobs=4,
        verbose=1,
        ax=ax
        )
ax.set_title(f'Learning Curve for binary {clf}')
plt.show()
plt.savefig(f'{fig_path}{today}_binary_lda_logocv_learning_curve.pdf')
end=datetime.now()
print(f'Learning curve complete at {end}, taking {end-start}')

start=datetime.now()
print(f'Now performing Leave-one-out cross validation with {len(groups.unique())} iterations, starting at {start}')
cv_score = cross_val_score(
        clf,
        X = X,
        y = y,
        groups = groups,
        cv = logocv
        )
end=datetime.now()
print(f'Finished at {end}, taking {end-start}')

print(f'Cross validation scores show mean of {np.mean(cv_score)}, standard deviation {np.std(cv_score)}')

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=0
        )

clf.fit(X_train, y_train)
'''
'''
start = datetime.now()
print(f'Starting Cross validated plotting at {start}')
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))
for fold, (train, test) in enumerate(cv.split(X, y)):
    print(f'Plotting fold {fold}')
    clf.fit(X.loc[train], y.loc[train])
    viz = RocCurveDisplay.from_estimator(
        clf,
        X.loc[test],
        y.loc[test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

mean_tpr = np.nanmean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label 'Tumour')",
)
ax.axis("square")
ax.legend(loc="lower right")
plt.show()
plt.savefig(f'{fig_path}{today}_cv_lda_roc.pdf')
end = datetime.now()
print(f'Plotting complete at {end}, taking {end-start}')


start = datetime.now()
print(f'Starting Cross validated precision recall plotting at {start}')
prc = []
av_prc= []
mean_recall = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))
for fold, (train, test) in enumerate(cv.split(X, y)):
    print(f'Plotting fold {fold}')
    clf.fit(X.loc[train], y.loc[train])
    viz = PrecisionRecallDisplay.from_estimator(
        clf,
        X.loc[test],
        y.loc[test],
        name=f"PR {fold}",
        alpha=0.8,
        lw=1,
        ax=ax,
    )
    interp_prc = np.interp(mean_recall, viz.recall, viz.precision)
    interp_prc[0] = 0.0
    prc.append(interp_prc)
    av_prc.append(viz.average_precision)
ax.set(
    xlabel="Recall",
    ylabel="Precision",
    title=f"Precision Recall Curves (10-fold cross-validation)",
)
ax.set_xlim(xmin=-0.05, xmax=1.05)
ax.set_ylim(ymin=-0.05, ymax=1.05)
#ax.axis("square")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(loc="lower left", fontsize='small')
#plt.show()
plt.savefig(f'{fig_path}{today}_cv_lda_pr_curve.pdf')
end = datetime.now()
print(f'Plotting complete at {end}, taking {end-start}')

print('Plotting binary LDA model')
bin_lda_plotter(bin_lda_data)
# now doing similar but with just the binary model
print('Plotting complete')



print('Refitting LDA with multiclass target')
y2 = data.path
clf.fit(X, y2)

X3 = clf.transform(X)
multi_lda_data = pd.DataFrame(X3)
multi_lda_data['Class'] = y2

print('Plotting multiclass LDA model in 2D and 3D')
lda_2d_plotter(multi_lda_data)
lda_3d_plotter(multi_lda_data)
print('Plotting complete')
'''


#####################
# FEATURE ELIMINATION
#####################

print('Starting the section on feature elimination, attempting to remove redundant, uninfromative and correlated features to speed up the model without compromising classification performance')

print('Creating correlation heatmap to visualise correlation in the data')
print('++WARNING++ This takes a really long time')
start = datetime.now()
heat_map_path = f'{fig_path}{today}_pp_correlation_heatmap.png'
print(f'Starting creating correlation matrix at {start}')
corr_matrix = X.corr(method = 'spearman').abs()

'''
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.set(font_scale = 1)
print(f'Starting plot generation at {datetime.now()}')
f, ax = plt.subplots(figsize=(7, 7))
# Make heatmap
sns.heatmap(corr_matrix, cmap= 'YlGnBu', square=True, vmin=-1, vmax=1, center=0,  mask=mask, ax = ax)
f.tight_layout()
# Save figure
f.savefig(heat_map_path, dpi=500)
end = datetime.now()
print(f'Figure saved to {heat_map_path}, that took {end-start}')
'''








