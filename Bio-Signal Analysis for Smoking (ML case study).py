#!/usr/bin/env python
# coding: utf-8

# # Vaishali_D_Bio-Signal Analysis for Smoking (ML Case Study)

# In[1]:


"""Demonstrate Logistic regression, Decision tree & Random forest and compare accuracy 
also do feature selection. Give conclusion which ML model is perfect for this case study."""


# In[1]:


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
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df=pd.read_csv(r"C:\Users\vaish\Downloads\smoking.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# # Logistic Regression

# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated()


# In[7]:


df[df.duplicated()]


# In[8]:


# Target Variable:
print(df['smoking'].value_counts())


# In[9]:


df = df.drop('ID', axis=1)
df.head()


# In[10]:


from sklearn.feature_selection import SelectKBest

skb=SelectKBest(k=8)
x = df.select_dtypes(include=['int64', 'float64'])
y=df.smoking

mod1=skb.fit(x,y)

d1={'Factor':x.columns,'Scores':mod1.scores_}
mydf=pd.DataFrame(d1)
print(mydf)


# In[11]:


from scipy.stats import pointbiserialr


# Step 1: Select only numeric columns excluding the binary 'Smoking_binary'
numeric_cols = df.select_dtypes(include='number').columns.drop('smoking')

# Step 3: Calculate point-biserial correlation between Smoking_binary and all numerical columns
results = []

for col in numeric_cols:
    corr, p_value = pointbiserialr(df['smoking'], df[col])
    results.append({'Variable': col, 'Correlation': corr, 'P-value': p_value})

# Convert results to DataFrame for easy viewing
corr_df = pd.DataFrame(results)

# Optional: Sort by absolute correlation
corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by='Abs_Correlation', ascending=False)

# Display the correlation table
print(corr_df[['Variable', 'Correlation', 'P-value']])



# In[12]:


df_object=df.select_dtypes(include='object').columns
df_object


# In[13]:


selected_columns = ['hemoglobin', 'height(cm)', 'weight(kg)', 'triglyceride', 'Gtp', 'waist(cm)',
                    'serum creatinine', 'HDL', 'age', 'relaxation', 'dental caries',
                    'fasting blood sugar']
df1 = df[selected_columns]
print(df1.head())


# In[14]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()

vif_data['Columns'] = df1.columns

res = []

for ctr in range(df1.shape[1]):
    res.append(variance_inflation_factor(df1.values,ctr))

vif_data['VIF'] = res
vif_data


# In[15]:


df = df.drop(['height(cm)', 'weight(kg)', 'waist(cm)', 'hemoglobin'], axis=1)
df.head()


# In[16]:


df = df.drop(['eyesight(left)', 'eyesight(right)', 'hearing(left)', 'systolic','Cholesterol','LDL','Urine protein','AST','ALT',], axis=1)
df.head()


# In[17]:


df = df.drop([ 'hearing(right)'], axis=1)
df.head()


# In[18]:


gender_dummy = pd.get_dummies(df.gender, drop_first=True,prefix='gender').astype(int)
df = df.drop('gender', axis=1)
df = pd.concat([df,gender_dummy], axis=1)


# In[19]:


oral_dummy = pd.get_dummies(df.oral, drop_first=True,prefix='oral').astype(int)
df = df.drop('oral', axis=1)
df = pd.concat([df,oral_dummy], axis=1)


# In[20]:


tartar_dummy = pd.get_dummies(df.tartar, drop_first=True,prefix='tartar').astype(int)
df = df.drop('tartar', axis=1)
df = pd.concat([df,tartar_dummy], axis=1)


# In[21]:


df.head()


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('smoking',axis=1), df.smoking, test_size=0.2, random_state=20)
x_train1 = sm.add_constant(x_train,prepend=False)
mod1 = sm.Logit(y_train,x_train1).fit()
print(mod1.summary())


# In[23]:


# Prepare test set
x_test_selected_const = sm.add_constant(x_test)

# Predict probabilities
y_pred_probs = mod1.predict(x_test_selected_const)

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


# In[24]:


x_test1 = sm.add_constant(x_test, prepend=False)

preds = mod1.predict(x_test1)

y_pred=[]
for ctr in preds:
    if ctr>=0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred))


# In[25]:


print(classification_report(y_test, y_pred))


# # Decision Tree

# In[35]:


dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_pred1 = dtc.predict(x_train)
y_pred2 = dtc.predict(x_test)
                                                    
from sklearn.metrics import classification_report

print(classification_report(y_train,y_pred1))
print(classification_report(y_test,y_pred2))


# In[36]:


dtc=DecisionTreeClassifier(criterion='entropy') 


# In[37]:


param = {'criterion':['gini','entropy','log_loss'],
         'max_depth':[2,3,4,5,6,7,8,9,10],
         'min_samples_split':[1,2,3,4,5,6,7],
         'min_samples_leaf':[1,2,3,4,5,6]
         }

from sklearn.model_selection import GridSearchCV

grd = GridSearchCV(dtc, param,cv=5,scoring='accuracy')
grd.fit(x_train,y_train)

grd.best_params_


# In[38]:


dtc = DecisionTreeClassifier(criterion='gini',
 max_depth= 9,
 min_samples_leaf= 1,
 min_samples_split= 3)

dtc.fit(x_train,y_train)

y_pred1 = dtc.predict(x_train)
y_pred2 = dtc.predict(x_test)
                                                    
from sklearn.metrics import classification_report

print(classification_report(y_train,y_pred1))
print(classification_report(y_test,y_pred2)) 


# # Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

par2 =          {'n_estimators':[100],
                              #'criterion': ['entropy','gini']
                               'max_depth':[10,20],
                               'min_samples_leaf': [2],
                               'min_samples_split': [2],
                              }

grid1=GridSearchCV(rfc,param_grid=par2,cv=5,scoring='accuracy')
grid1.fit(x_train,y_train)                             
grid1.best_params_


# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


rfc = RandomForestClassifier(max_depth = 20,
 min_samples_leaf= 2,
 min_samples_split= 2,
 n_estimators= 100)

rfc.fit(x_train,y_train)

y_pred1 = rfc.predict(x_train)
y_pred2 = rfc.predict(x_test)


# In[41]:


y_pred1


# In[42]:


y_pred2


# In[29]:


print(rfc.score(x_train,y_train)) # training data accuracy
print(rfc.score(x_test,y_test))


# In[30]:


print(classification_report(y_test,y_pred2))


# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


rfc = RandomForestClassifier(max_depth =None,
 min_samples_leaf= 1,
 min_samples_split= 5,
 n_estimators= 150)

rfc.fit(x_train,y_train)

y_pred1 = rfc.predict(x_train)
y_pred2 = rfc.predict(x_test)


# In[33]:


print(rfc.score(x_train,y_train)) # training data accuracy
print(rfc.score(x_test,y_test))


# In[34]:


print(classification_report(y_test,y_pred2))


# # Random Forest performs significantly better than both Logistic Regression and Decision Tree, achieving an accuracy of 82%, which is a 7-8% improvement over the others.

# In[ ]:




