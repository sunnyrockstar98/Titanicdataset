# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp']
testset['Family_Size'] = testset['Parch'] + testset['SibSp']
dataset['Alone']  = dataset['Family_Size'].map(lambda x: 0 if x>0 else 1)
testset['Alone']  = testset['Family_Size'].map(lambda x: 0 if x>0 else 1)


X = dataset.iloc[:,[2,4,5,6,7,9,12,13]].values
y = dataset.iloc[:, 1].values
X_testf = testset.iloc[:,[1,3,4,5,6,8,11,12]].values
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, [0,2,3,4,5]])
X[:, [0,2,3,4,5]] = imputer.transform(X[:,[0,2,3,4,5]])

imputer1 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer1 = imputer1.fit(X_testf[:, [0,2,3,4,5]])
X_testf[:, [0,2,3,4,5]] = imputer1.transform(X_testf[:,[0,2,3,4,5]])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X_testf[:, 1] = labelencoder_X1.fit_transform(X_testf[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_testf = onehotencoder.fit_transform(X_testf).toarray()
# Encoding the Dependent Variable

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
X_testf = X_testf[:, 1:]

'''
#Building optima backward elimination model - using both p-vals and adj - R
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((891,1)).astype(int),values = X, axis = 1)
X_testf = np.append(arr = np.ones((418,1)).astype(int),values = X_testf, axis = 1)
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((len(X[:,0]),6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, 0:]
X_Modeled = backwardElimination(X_opt, SL)
X_opttestf = X_testf[:, 0:]
X_Modeledtestf = backwardElimination(X_opttestf, SL)'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_testf = sc.transform(X_testf)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0,C=1,gamma = 0.1)
classifier.fit(X, y)
# Predicting the Test set results
y_predf = classifier.predict(X_testf)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
accuracies.mean()

#
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 1.10, 1.20, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]}]
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           )
#grid_search = grid_search.fit(X, y)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_