#!/usr/bin/env python
# coding: utf-8

# # Vaishali_Deshmukh
# #  Simple Linear Regression 
# #  Prediction of Salary using Year of Experience

# In[6]:


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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats import weightstats as ssw
import statsmodels.stats.multicomp
from statsmodels.stats import proportion as ssp
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


# # Data Loading

# In[7]:


df=pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\CDAC Big Data Anyaltics\Stats\ML_Case Studies\data.csv")
df.head()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# # EDA

# In[12]:


plt.scatter(df.YearsExperience , df.Salary)
plt.ylabel('Salary')
plt.xlabel('Years_of_Experience')

'''Using Scatter diagram we can see that as Years of experienece incarese salary also increase, so positive correlation is there between this two
Variable.'''


# # Inferential Test

# In[13]:


correlation=df['YearsExperience'].corr(df['Salary'],method='pearson')
correlation


# # Value of Correlation Coefficient is 0.9782 So there is strong positive correlation Between Year of Experience and Salary.

# # Model Fitting:

# In[14]:


x_train,x_test,y_train,y_test=train_test_split(df.YearsExperience,df.Salary,test_size=0.2) # 80% Data split that is 0.2.


# In[15]:


x_train1=sm.add_constant(x_train,prepend=False)


# In[16]:


mod=sm.OLS(y_train,x_train1).fit()


# In[17]:


print(mod.summary())


# Model Summary
# 
# Null Hypothesis (H₀): Salary is not influenced by Years of Experience.
# 
# Alternative Hypothesis (H₁): Salary is influenced by Years of Experience.
# 
# Based on the model output, the p-value for the variable YearsExperience is less than 0.05, indicating that the relationship is statistically significant. 
# Therefore, we reject the null hypothesis (H₀) and conclude that Salary is significantly influenced by Years of Experience.
# 
# The R-squared value is 0.953, which means that 95.3% of the variation in Salary is explained by Years of Experience.
# This indicates that the model provides a very good fit to the data.

# # Model Testing :

# In[18]:


x_test1 = sm.add_constant(x_test,prepend=False)
y_pred = mod.predict(x_test1)


# In[19]:


y_pred.head()


# # Answer :

# In[4]:


# Q1
experienced_employees = df[df['YearsExperience'] > 5]
employees_earning_60k = experienced_employees[experienced_employees['Salary'] >= 60000]
num_employees = len(employees_earning_60k)
print(num_employees)
# option: D


# In[5]:


#Q2
num_emp = df[(df['Salary'] >= 50000) & (df['Salary'] <= 80000)]
print(len(num_emp))
# option: B


# Q.3 Ans : Scatter Diagram Interpretation:
# Plot Shows Postive Correlation between YeasrsExperinece ans Salary col

# Q.4 Ans : The distribution of Years of Experience is positive skewed.

# Q.5 Ans : The Range of 'YearsExperience' and 'salary' Data is (9.4,84660)

# Q.6 Ans: The training data will consist of 80% of the samples from the total population.

# Q.7 The random state does not have any effect on the shape of the data.

# In[34]:


# Q.8
x_train,x_test,y_train,y_test=train_test_split(df.YearsExperience,df.Salary,test_size=0.4)
x_train1=sm.add_constant(x_train,prepend=False)
mod1=sm.OLS(y_train,x_train1).fit()
print(mod1.summary())


# Q.9 Reshape the data into two dimensional array.

# In[ ]:




