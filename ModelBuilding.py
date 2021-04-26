# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('eda_data.csv')

print (df.columns)

#choose relevant columns
df_model = df[['avg_salary','Title', 'Rating','Location', 'Employee Provided','Python', 'Excel', 'AWS', 'SQL',
       'Seniority', 'desc_len', 'min_salary', 'max_salary',
       'hourly']]

#get dummy data
df_dummy = pd.get_dummies(df_model)

#train test split
from sklearn.model_selection import train_test_split

x = df_dummy.drop('avg_salary', axis=1)
y = df_dummy.avg_salary.values

x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#multiple linear regression

import statsmodels.api as sm

x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
print(model.summary())


#lasso regression




#random forest

#tune models GridSearchCV

#test ensembles

