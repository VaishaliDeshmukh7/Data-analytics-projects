#!/usr/bin/env python
# coding: utf-8

# # Vaishali_Deshmukh
# # Logistic_reg_Credit_Worthiness

# In[ ]:


from ast import increment_lineno
import numpy as np
import pandas as pd
from numpy import random
from numpy.linalg import inv
import os
import scipy
from scipy import stats
from scipy.stats import skew,kurtosis
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import t
from statsmodels import stats
from statsmodels.stats import  weightstats as ssw
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats import proportion as ssp
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import f
from statsmodels.stats import rates
from statsmodels.stats.rates import test_poisson
from statsmodels.stats.rates import test_poisson_2indep
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from statsmodels import discrete
from statsmodels.discrete import discrete_model
from statsmodels.discrete.discrete_model import MNLogit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor


# In[5]:


df=pd.read_excel(r"C:\Users\vaish\Downloads\CreditWorthiness.xlsx", sheet_name='Data')
df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['creditScore_encoded'] = le.fit_transform(df['creditScore'])

df = df.drop('creditScore',axis=1)
df.head()


# In[10]:


for col in df.select_dtypes(include=['object', 'category']).columns:
    print(df.groupby(col)['creditScore_encoded'].mean().sort_values())
    print('-'*40)


# In[11]:


cat_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(int)
print(df_encoded.head())


# In[12]:


df_encoded.columns = df_encoded.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)


# In[13]:


print(df_encoded.head())


# In[14]:


x = df_encoded.drop('creditScore_encoded', axis=1)
y = df_encoded['creditScore_encoded']
x_const = sm.add_constant(x)
logit_model = sm.Logit(y, x_const)
result = logit_model.fit()
print(result.summary())


# In[15]:


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x_const, y, test_size=0.3, random_state=42)
logit_model = sm.Logit(y_train, x_train)
result = logit_model.fit()

y_pred_prob = result.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[16]:


selected_features = [
    'Cdur',
    'Camt',
    'InRate',
    'Cbal_Rs_2000',
    'Cbal_no_checking_account',
    'Chist_all_settled_till_now',
    'Chist_dues_not_paid_earlier',
    'Cpur_new_vehicle',
    'Cpur_second_hand_vehicle',
    'MSG_single_male',
    'Oparties_yes_guarantor',
    'Rdur_less_than_a_year',
    'inPlans_none',
    'foreign_yes'
]

target = 'creditScore_encoded'


# In[17]:


import statsmodels.api as sm
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]
x_train_selected_const = sm.add_constant(x_train_selected)
logit_model_selected = sm.Logit(y_train, x_train_selected_const).fit()
print(logit_model_selected.summary())


# In[18]:


# Prepare test set
x_test_selected_const = sm.add_constant(x_test_selected)

# Predict probabilities
y_pred_probs = logit_model_selected.predict(x_test_selected_const)

# Threshold at 0.5
y_pred_class = (y_pred_probs >= 0.5).astype(int)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_probs)
cm = confusion_matrix(y_test, y_pred_class)

print("Accuracy:", accuracy)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)


# In[ ]:




