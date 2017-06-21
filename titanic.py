import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble
import matplotlib.pyplot as plt


# read data
data = pd.read_csv("train.csv", index_col=0)
X = data.drop('Survived', axis=1)
y = data['Survived']


## data prep
# Name
titles_series = X['Name'].str.extract('(\w+)(?=\.)', expand=False)
X_titles = pd.get_dummies(titles_series, prefix='Title')
X = pd.concat((X, X_titles), axis=1)

# Sex
X['Sex'] = X['Sex'].map({'male' : 0, 'female' : 1})

# Age
X['Age'] = X['Age'].fillna(X['Age'].mean())

# SibSp + Parch
X['Family_Size'] = X['SibSp'] + X['Parch']

# Cabin
cabin_series = X['Cabin'].str.extract('([a-zA-Z])', expand=False)
X_cabin = pd.get_dummies(cabin_series, prefix='Cabin')
X_cabin = X_cabin.drop('Cabin_T', axis=1)

# Embarked
X_embarked = pd.get_dummies(X['Embarked'], prefix='Embarked')


X = pd.concat((X.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], axis=1), X_titles, X_embarked, X_cabin), axis=1)
X = pd.DataFrame(X, dtype=float)
X = (X - X.mean()) / X.std()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)


## training
best_result = 1

# svc
svc = SVC()
svc.fit(X_train, y_train)
err_test  = pd.np.mean(y_test  != svc.predict(X_test))
print("SVC (rbf)", err_test)
if (err_test < best_result): best_result = err_test

svc = SVC(kernel='linear', C=0.0428133239872)
svc.fit(X_train, y_train)
err_test  = pd.np.mean(y_test  != svc.predict(X_test))
print("SVC (linear)", err_test)
if (err_test < best_result): best_result = err_test

svc = SVC(kernel='poly', C=14, degree=2)
svc.fit(X_train, y_train)
err_test  = pd.np.mean(y_test  != svc.predict(X_test))
print("SVC (poly)", err_test)
if (err_test < best_result): best_result = err_test

# rf
rf = ensemble.RandomForestClassifier(n_estimators=32, random_state=1234)
rf.fit(X_train, y_train)
err_test  = pd.np.mean(y_test  != rf.predict(X_test))
print("RF1", err_test)
if (err_test < best_result): best_result = err_test

# gbt
gbt = ensemble.GradientBoostingClassifier(random_state=1)
gbt.fit(X_train, y_train)
err_test = pd.np.mean(y_test != gbt.predict(X_test))
print("GBT1", err_test)
if (err_test < best_result): best_result = err_test