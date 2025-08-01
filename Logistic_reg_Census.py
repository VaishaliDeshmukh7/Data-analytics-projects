#!/usr/bin/env python
# coding: utf-8

# # Vaishali_Deshmukh
# # Logistic_reg_Census

# In[ ]:


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


# In[2]:


df=pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\CDAC Big Data Anyaltics\Stats\ML_Case Studies\census_income.csv")
df.head()


# In[3]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


for col in df.select_dtypes(include = 'object').columns.tolist():
  print(col,":", np.unique(df[col]))


# In[9]:


df.select_dtypes(include = 'object').columns.tolist()


# In[10]:


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

categorical_cols.remove('annual_income')

df['annual_income'] = df['annual_income'].map({'<=50K': 0, '>50K': 1})

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_encoded = df_encoded.astype(int)


# In[11]:


df_encoded.head()


# In[12]:


from sklearn.model_selection import train_test_split

X = df_encoded.drop('annual_income', axis=1)
y = df_encoded['annual_income']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


# In[13]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)


# In[14]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


y_pred_prob = model.predict_proba(X_test)[:, 1]

y_pred_class = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred_class)

print("Accuracy:", accuracy)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)


# In[15]:


feature_names = X_train.columns


# In[16]:


coefficients = model.coef_[0]


# In[17]:


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})


# In[18]:


importance_df['Odds_Ratio'] = np.exp(importance_df['Coefficient'])


# In[19]:


importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)


# In[20]:





# Education, marital status, occupation, and relationship status emerge as strong predictors of income level — as anticipated based on socioeconomic theory.
# 
# Individuals in executive/professional roles and those with higher education levels have a significantly higher likelihood of earning above the income threshold.
# 
# Conversely, being never-married or working in the service sector is associated with a notably lower probability of high income.
# 
# Interestingly, missing or 'Unknown' values also appear to carry informative patterns, potentially reflecting underlying socioeconomic or demographic characteristics.
