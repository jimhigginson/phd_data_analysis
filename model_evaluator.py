'''
in this script I'm going to create a function (or class??) that takes a pickled rfecv model and the dataset from which it was created, and generate a report of all the relevant metrics.
'''

from datetime import date
import pickle



today = date.today()

def title(string):
    banner = 40*'#'
    print(f'\n{banner}\n'+string+f'\n{banner}\n')


# import the models here
model_path = './models/'
binary_lda_file = 'binary_lda_rfecv'
binary_lda = pickle.load(open(f'{model_path}{today}_{binary_lda_file}.pkl','rb'))




######################
# Report starts here #
######################

title(f'This report was generated on {today}')
print(f'The first model is {binary_lda_file}')
print(f'this model used the {binary_lda.scoring} scoring model to optimise the number of features')
