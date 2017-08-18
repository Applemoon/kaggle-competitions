import pandas as pd
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#from sklearn import ensemble
#from sklearn.model_selection import StratifiedKFold
#from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
#import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv", index_col=0)
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
train_len = X.shape[0]

X_test = pd.read_csv("test.csv", index_col=0)
X = pd.concat((X, X_test))


## data prep
# Name
titles_series = X['Name'].str.extract('(\w+)(?=\.)', expand=False)
X_titles = pd.get_dummies(titles_series, prefix='Title')
X = pd.concat((X, X_titles), axis=1)

# Cabin
cabin_series = X['Cabin'].str.extract('([a-zA-Z])', expand=False)
X_cabin = pd.get_dummies(cabin_series, prefix='Cabin')
X_cabin = X_cabin.drop('Cabin_T', axis=1)

# Embarked
X_embarked = pd.get_dummies(X['Embarked'], prefix='Embarked')

X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
X['Family_Size'] = X['SibSp'] + X['Parch']
X['Sex'] = X['Sex'].map({'male' : 0, 'female' : 1})


X = pd.concat((X.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], axis=1), X_titles, X_embarked, X_cabin), axis=1)
X = pd.DataFrame(X, dtype=float)
X = (X - X.mean()) / X.std()


## Split 1
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

## Split 2
X_train = X[:train_len]
y_train = y
X_test = X[train_len:]


## training
# svc
svc = SVC()
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)
prediction_df = pd.DataFrame(index=X_test.index, data=prediction, columns=['Survived'])
