import pandas as pd
import numpy as np


# read data
data = pd.read_csv("train.csv", index_col=0)
X = data.drop('Survived', axis=1)
y = data.Survived


## подготовка данных
# Name
titles_series = X.Name.str.extract('(\w+)(?=\.)')
X_titles = pd.get_dummies(titles_series, prefix='Title')
X = pd.concat((X, X_titles), axis=1)

# Sex
X.Sex = X.Sex.map({'male' : 0, 'female' : 1})

# Age
X.Age = X.Age.fillna(X.Age.mean())

# Cabin
# TODO

# Embarked
X_embarked = pd.get_dummies(X.Embarked, prefix='Embarked')


X = pd.concat((X.drop(['Name', 'Embarked', 'Ticket'], axis=1), X_titles, X_embarked), axis=1)
