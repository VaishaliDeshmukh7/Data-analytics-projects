#!/usr/bin/env python
# coding: utf-8

# # Vaishali_Deshmukh
# # MLR_House_Price_Prediction

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


# # Data Set loading

# In[3]:


df= pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\CDAC Big Data Anyaltics\Stats\ML_Case Studies\housing.csv")
df.head()


# In[62]:


df.shape


# In[63]:


df.info()


# In[64]:


df= df.drop(['id','date','zipcode'],axis=1)


# In[65]:


df.head()


# In[66]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
sns.boxplot(x=df['price'])
plt.title('Boxplot of House Prices')
plt.show()


# In[31]:


sns.pairplot(df[['price', 'sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']])
plt.show()


# In[67]:


correlation=df['sqft_living'].corr(df['price'],method='pearson')
correlation


# In[68]:


correlation=df['sqft_living15'].corr(df['price'],method='pearson')
correlation


# In[69]:


correlation=df['sqft_lot'].corr(df['price'],method='pearson')
correlation


# In[70]:


print(df.columns)


# In[71]:


selected_cols = ['price', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'yr_built',
       'yr_renovated', 'sqft_living15', 'sqft_lot15','lat','long']
df_selected = df[selected_cols]

corr_matrix = df_selected.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Variables')
plt.show()


# In[13]:


sns.pairplot(df[['price','floors','sqft_above','sqft_basement']])
plt.show()


# In[72]:


import pandas as pd
cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

# Compute correlation of each with 'price'
correlations = df[cols + ['price']].corr()['price'].drop('price')

# Display
print(correlations.sort_values(ascending=False))


# In[73]:


selected_columns = ['price', 'sqft_living', 'grade', 'sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','lat']
df_final= df[selected_columns]


# In[ ]:


# Using Categorical Regression model


# In[74]:


selected_columns = ['price', 'sqft_living', 'grade', 'sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','lat']
df_final= df[selected_columns]
df_final.head()


# In[75]:


gen_dummy = pd.get_dummies(df_final['grade'],drop_first=True,prefix='grade').astype(int) # Create Dummy Col
df_final= df_final.drop('grade', axis=1) # Remove original col
df_final = pd.concat([df_final,gen_dummy], axis=1)  # add dummy col in df


view_dummy = pd.get_dummies(df_final['view'],drop_first=True,prefix='view').astype(int) # Create Dummy Col
df_final= df_final.drop('view', axis=1) # Remove original col
df_final = pd.concat([df_final,view_dummy], axis=1)  # add dummy col in df

x_train, x_test, y_train, y_test = train_test_split(df_final.drop('price',axis=1), df_final.price, test_size=0.2, random_state=20)
x_train1 = sm.add_constant(x_train,prepend=False)
mod1 = sm.OLS(y_train,x_train1).fit()
print(mod1.summary())


# In[ ]:


# After removing Outliers:


# In[85]:


selected_columns = ['price', 'sqft_living', 'grade', 'sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','lat']
df_final1= df[selected_columns]


# In[87]:


df_final1.head()


# In[88]:


df_final1.shape


# In[94]:


df_final1.isnull().sum()


# In[96]:


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Remove rows with outliers in price
df_final1 = df_final1[(df_final1['price'] >= lower_bound) & (df_final1['price'] <= upper_bound)]


# In[97]:


df_final1.shape


# In[98]:


df_final1.head()


# In[99]:


gen_dummy = pd.get_dummies(df_final1['grade'],drop_first=True,prefix='grade').astype(int) # Create Dummy Col
df_final1= df_final1.drop('grade', axis=1) # Remove original col
df_final1= pd.concat([df_final1,gen_dummy], axis=1)  # add dummy col in df


view_dummy = pd.get_dummies(df_final1['view'],drop_first=True,prefix='view').astype(int) # Create Dummy Col
df_final1= df_final1.drop('view', axis=1) # Remove original col
df_final1= pd.concat([df_final1,view_dummy], axis=1)  # add dummy col in df

x_train, x_test, y_train, y_test = train_test_split(df_final1.drop('price',axis=1), df_final1.price, test_size=0.2, random_state=20)
x_train1 = sm.add_constant(x_train,prepend=False)
mod2 = sm.OLS(y_train,x_train1).fit()
print(mod2.summary())


# # Ans :

# In[5]:


# Q.10:
waterfront_count = df[df['waterfront'] == 1].shape[0]
waterfront_count
# a)


# In[7]:


# Q.11:
two_floors_count = df[df['floors'] == 2].shape[0]
two_floors_count
# b)


# In[8]:


# Q.12: 
before_1960_waterfront = df[(df['yr_built'] < 1960) & (df['waterfront'] == 1)].shape[0]
before_1960_waterfront
# a)


# In[9]:


# Q.13:
expensive_house_price = df[df['bathrooms'] > 4]['price'].max()
expensive_house_price
# a)


# In[ ]:


Q.14:c)


# In[ ]:


Q.15:a)


# In[ ]:


Q.16:a) 


# In[ ]:


Q.17: a)The model is 38% accurate and shows poor efficiency.
d) high difference between observed and fitted values.


# In[ ]:


Q.18: c) Insingificance in presence of other variables.


# In[ ]:


Q.19: a) High multicollinerity

