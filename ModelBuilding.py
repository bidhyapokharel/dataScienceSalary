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


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

ln = LinearRegression()
ln.fit(x_train, y_train)

np.mean(cross_val_score(ln,x_train,y_train,scoring='neg_mean_absolute_error',cv=3))

#lasso regression
ln_l = Lasso(alpha=.13)
ln_l.fit(x_train,y_train)
np.mean(cross_val_score(ln_l, x_train, y_train, scoring='neg_mean_absolute_error',cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/10)
    lms = Lasso(alpha=i/10)
    error.append(np.mean(cross_val_score(ln_l, x_train, y_train, scoring='neg_mean_absolute_error',cv=3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns=['alpha','error'])
df_err[df_err == max(df_err)]

#random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

np.mean(cross_val_score(rf,x_train,y_train,scoring='neg_mean_absolute_error',cv=3))


#tune models GridSearchCV

from sklearn.model_selection import GridSearchCV

parameters = {'n-estimators':range(10,300,10), 'criterion':('mse','mae'),'max_features':('auto','sqrt','Log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(x_train,y_train)
print(gs.best_score_)
print(gs.best_estimator_)

#test ensembles

tpred_lm = ln.predict(x_test)
tpred_lml = ln_l.predict(x_test)
tpred_rf = gs.best_estimator_.predict(x_test)


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,rf)
mean_absolute_error(y_test,ln)
mean_absolute_error(y_test, ln_l)

print(mean_absolute_error(y_test,(tpred_lm + tpred_rf / 2)))