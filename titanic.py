import pandas as pd
import numpy as np


# read data
data = pd.read_csv("train.csv", index_col=0)
X = data.drop('Survived', axis=1)
y = data.Survived


## подготовка данных
# Name
titles_series = X.Name.str.extract('(\w+)(?=\.)', expand=False)
X_titles = pd.get_dummies(titles_series, prefix='Title')
X = pd.concat((X, X_titles), axis=1)

# Sex
X.Sex = X.Sex.map({'male' : 0, 'female' : 1})

# Age
X.Age = X.Age.fillna(X.Age.mean())

# SibSp + Parch
# TODO

# Cabin
cabin_series = X.Cabin.str.extract('([a-zA-Z])', expand=False)
X_cabin = pd.get_dummies(cabin_series, prefix='Cabin')
X_cabin = X_cabin.drop('Cabin_T', axis=1)

# Embarked
X_embarked = pd.get_dummies(X.Embarked, prefix='Embarked')


X = pd.concat((X.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], axis=1), X_titles, X_embarked, X_cabin), axis=1)
