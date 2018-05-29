# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp']
dataset['Age'].fillna(dataset['Age'].mode()[0],inplace=True)
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
testset['Family_Size'] = testset['Parch'] + testset['SibSp']

testset['Age'].fillna(testset['Age'].mode()[0],inplace=True)
testset['Embarked'].fillna(testset['Embarked'].mode()[0],inplace=True)
testset['Pclass'].fillna(testset['Pclass'].mode()[0],inplace=True)
testset['Sex'].fillna(testset['Sex'].mode()[0],inplace=True)
testset['SibSp'].fillna(testset['SibSp'].mode()[0],inplace=True)
testset['Parch'].fillna(testset['Parch'].mode()[0],inplace=True)
testset['Fare'].fillna(testset['Fare'].mode()[0],inplace=True)
dataset['Alone']  = dataset['Family_Size'].map(lambda x: 0 if x>0 else 1)
testset['Alone']  = testset['Family_Size'].map(lambda x: 0 if x>0 else 1)


X = dataset.iloc[:,[2,4,5,6,7,9,12,13]].values
y = dataset.iloc[:, 1].values
X_testf = testset.iloc[:,[1,3,4,5,6,8,11,12]].values


  

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
#labelencoder_X1 = LabelEncoder()
#X[:, 6] = labelencoder_X1.fit_transform(X[:, 6])
#onehotencoder = OneHotEncoder(categorical_features = [6])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

labelencoder_X_testf = LabelEncoder()
X_testf[:, 1] = labelencoder_X_testf.fit_transform(X_testf[:, 1])
#labelencoder_X_testf1 = LabelEncoder()
#X_testf[:, 6] = labelencoder_X_testf1.fit_transform(X_testf[:, 6])
#onehotencoder = OneHotEncoder(categorical_features = [6])
#X_testf = onehotencoder.fit_transform(X_testf).toarray()
#X_testf = X_testf[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_testf = sc.transform(X_testf)


# Fitting Kernel SVM to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
# Predicting the Test set results
y_predf = classifier.predict(X_testf)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
accuracies.mean()