#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[ ]:


#  1. Load the dataset and display the structure (rows, columns, data types).


# In[4]:


df=pd.read_csv(r"C:\Users\vaish\Downloads\used_cars_data.csv")
df.head()


# In[5]:


df.shape


# In[7]:


df.dtypes


# In[6]:


# 2. Identify and handle missing values across the dataset


# In[8]:


df.isnull().sum()


# In[9]:


# Clean and convert 'Mileage', 'Engine', and 'Power' to numeric


# In[10]:


df['Mileage'] = df['Mileage'].str.extract('(\d+.\d+)').astype(float)
df['Engine'] = df['Engine'].str.extract('(\d+)').astype(float)
df['Power'] = df['Power'].str.extract('(\d+.\d+)').astype(float)


# In[11]:


df['Mileage'].fillna(df['Mileage'].median(), inplace=True)
df['Engine'].fillna(df['Engine'].median(), inplace=True)
df['Power'].fillna(df['Power'].median(), inplace=True)
df['Seats'].fillna(df['Seats'].median(), inplace=True)


# In[12]:


df.isnull().sum()


# In[13]:


df.dropna(subset=['Price'], inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


# 1. Check total number of duplicate rows
duplicate_count = df.duplicated().sum()
duplicate_count


# In[17]:


#  Descriptive Analysis


# In[17]:


# 1. Generate summary statistics (mean, median, std. dev., etc.) for numeric features like 
 #Price, Mileage, Power, Engine, Kilometers_Driven.


# In[18]:


# List of numeric columns to summarize
num_cols = ['Price', 'Mileage', 'Power', 'Engine', 'Kilometers_Driven']


# In[19]:


# Generate summary statistics
summary = df[num_cols].describe().T  # Transpose for better readability
summary['median'] = df[num_cols].median()
summary = summary[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]


# In[20]:


print(summary)


# In[ ]:


# 2. Analyze the distribution of categorical features like Fuel_Type, Transmission, Owner_Type, and Location.


# In[20]:


print(df['Fuel_Type'].value_counts())


# In[21]:


# Bar Plot
sns.countplot(data=df, x='Fuel_Type', order=df['Fuel_Type'].value_counts().index)


# In[22]:


print(df['Transmission'].value_counts())


# In[23]:


# Bar Plot
sns.countplot(data=df, x='Transmission', order=df['Transmission'].value_counts().index)


# In[24]:


print(df['Owner_Type'].value_counts())


# In[25]:


# Bar Plot
sns.countplot(data=df, x='Owner_Type', order=df['Owner_Type'].value_counts().index)


# In[26]:


print(df['Location'].value_counts())


# In[27]:


# Bar Plot
sns.countplot(data=df, x='Location', order=df['Location'].value_counts().index)


# In[ ]:


#  Visual Explorations


# In[30]:


#3. Create histograms and boxplots for numerical features like Price, Engine, and Mileage.


# In[28]:


# Price
sns.histplot(data=df, x='Price', bins=50, kde=True, color='skyblue')
plt.title('Histogram of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[29]:


sns.boxplot(data=df, x='Price', color='lightgreen')
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.show()


# In[30]:


# Engine
sns.histplot(data=df, x='Engine', bins=50, kde=True, color='salmon')
plt.title('Histogram of Engine Size')
plt.xlabel('Engine (cc)')
plt.ylabel('Frequency')
plt.show()


# In[31]:


sns.boxplot(data=df, x='Engine', color='orange')
plt.title('Boxplot of Engine Size')
plt.xlabel('Engine (cc)')
plt.show()


# In[32]:


# Mileage
sns.histplot(data=df, x='Mileage', bins=50, kde=True, color='orchid')
plt.title('Histogram of Mileage')
plt.xlabel('Mileage (kmpl or km/kg)')
plt.ylabel('Frequency')
plt.show()


# In[33]:


sns.boxplot(data=df, x='Mileage', color='lightblue')
plt.title('Boxplot of Mileage')
plt.xlabel('Mileage (kmpl or km/kg)')
plt.show()


# In[34]:


#Visualize the trend of average car prices over Year of manufacture.
 
# Step 1: Group by Year and calculate average Price
yearly_price_trend = df.groupby('Year')['Price'].mean().reset_index()

# Step 2: Sort by Year (optional, for cleaner plot)
yearly_price_trend = yearly_price_trend.sort_values('Year')

# Step 3: Plot the trend
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_price_trend, x='Year', y='Price', marker='o', color='teal')
plt.title('Average Car Price Trend Over Year of Manufacture')
plt.xlabel('Year of Manufacture')
plt.ylabel('Average Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[35]:


# pare price distributions across different Locations and Fuel_Types.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='Location', y='Price', palette='Set2')
plt.title('Price Distribution Across Locations')
plt.xlabel('Location')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[36]:


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Fuel_Type', y='Price', palette='pastel')
plt.title('Price Distribution Across Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.tight_layout()
plt.show()


# In[ ]:


#  Correlation and Feature Relationships


# In[37]:


# 7. Compute a correlation matrix for numerical columns and visualize it using a heatmap.
# Ensure columns are numeric
num_cols = ['Price', 'Mileage', 'Engine', 'Power', 'Kilometers_Driven']
corr_matrix = df[num_cols].corr()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# In[41]:


# 8. Scatter Plots: Price vs Key Numerical Variables
# Price vs Mileage
sns.scatterplot(data=df, x='Mileage', y='Price', color='blue')
plt.title('Price vs Mileage')
plt.show()

# Price vs Engine
sns.scatterplot(data=df, x='Engine', y='Price', color='green')
plt.title('Price vs Engine')
plt.show()

# Price vs Power
sns.scatterplot(data=df, x='Power', y='Price', color='purple')
plt.title('Price vs Power')
plt.show()

# Price vs Kilometers_Driven
sns.scatterplot(data=df, x='Kilometers_Driven', y='Price', color='orange')
plt.title('Price vs Kilometers Driven')
plt.show()


# In[40]:


# 9. Multivariate Analysis: Transmission & Fuel_Type → Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Transmission', y='Price', hue='Fuel_Type', palette='Set3')
plt.title('Effect of Transmission and Fuel Type on Price')
plt.show()


# In[42]:


plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Transmission', y='Price', hue='Fuel_Type', estimator='mean', ci='sd', palette='pastel')
plt.title('Average Price by Transmission and Fuel Type')
plt.show()


# In[ ]:


# Outliers and Data Quality


# In[43]:


# 10. Boxplots to Identify Outliers in Price, Engine, and Power
import matplotlib.pyplot as plt
import seaborn as sns

for col in ['Price', 'Engine', 'Power']:
   plt.figure(figsize=(8, 4))
   sns.boxplot(x=df[col], color='skyblue')
   plt.title(f'Boxplot for {col}')
   plt.xlabel(col)
   plt.show()


# In[44]:


# 11. Identify Features with Possible Inconsistencies
# Check sample of problematic entries
print("Unique Power examples:", df['Power'].dropna().unique()[:10])
print("Unique Mileage examples:", df['Mileage'].dropna().unique()[:10])


# In[46]:


# Strip text units and convert to numeric:

# Power: remove ' bhp' and convert to float
df['Power'] = df['Power'].astype(str).str.replace(' bhp', '', regex=False)
df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

# Engine: remove ' CC' and convert to float
df['Engine'] = df['Engine'].astype(str).str.replace(' CC', '', regex=False)
df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')

# Mileage: remove both ' kmpl' and ' km/kg', then convert
df['Mileage'] = df['Mileage'].astype(str).str.replace(' kmpl', '', regex=False)
df['Mileage'] = df['Mileage'].str.replace(' km/kg', '', regex=False)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')


# In[47]:


df[['Power', 'Engine', 'Mileage']].info()
df[['Power', 'Engine', 'Mileage']].describe()


# In[48]:


# 12. Analyze New_Price Feature and Its Missing Data
# Step 1: Check percentage missing:
missing_percentage = df['New_Price'].isnull().mean() * 100
print(f"New_Price is missing in {missing_percentage:.2f}% of the rows.")


# In[ ]:




