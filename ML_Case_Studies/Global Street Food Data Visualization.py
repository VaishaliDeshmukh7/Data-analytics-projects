#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
import matplotlib.cm as cm


# In[ ]:


# B1. Data Loading and Exploration


# In[6]:


df=pd.read_csv(r"C:\Users\vaish\Downloads\global_street_food.csv")
df.head()


# In[27]:


df.shape


# In[7]:


df.columns


# In[8]:


df.isnull().sum()


# In[7]:


# B2. Simple Bar Chart


# In[8]:


# Create a vertical bar chart showing the count of dishes by country:
# Create a vertical bar chart showing the count of dishes by country:
# Use only the top 6 countries
# Add title "Street Food Dishes by Country"
# Label x and y axes
# Use blue color for all bars


# In[12]:


country_counts=df['Country'].value_counts().head(6)
country_counts


# In[15]:


plt.figure(figsize=(10,6))
plt.title("Street Food Dishes by Country")
plt.xlabel("Country")
plt.ylabel("Dish Name")
country_counts.plot(kind='bar',color='blue')


# In[ ]:


# B3. Basic Histogram (8 points)
'''Create a histogram of dish prices:
Use 20 bins
Set title "Distribution of Street Food Prices"
Add xlabel "Price (USD)" and ylabel "Frequency"
Use green color'''


# In[16]:


plt.figure(figsize=(10, 6))
plt.hist(df['Typical Price (USD)'], bins=20, color='green', edgecolor='black')
plt.title("Distribution of Street Food Prices")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")


# In[ ]:


# B4. Simple Pie Chart (9 points)
'''Create a pie chart for vegetarian vs non-vegetarian dishes:
Show percentages on the chart
Use colors: green for "Yes", red for "No"
Add title "Vegetarian vs Non-Vegetarian Street Food" '''


# In[18]:


veg_counts = df['Vegetarian'].value_counts()
veg_counts


# In[21]:


colors = ['green', 'red']
labels = ['Yes', 'No']
plt.figure(figsize=(5, 5))
plt.pie(veg_counts, labels=labels,colors=colors)
plt.title("Vegetarian vs Non-Vegetarian Street Food")


# In[ ]:


# INTERMEDIATE LEVEL (Comfortable with Pandas/Matplotlib)
'''I1. Enhanced Bar Chart with Styling (12 points)
Create a horizontal bar chart of cooking methods:
Show all cooking methods
Sort bars by count (highest to lowest)
Use different colors for each bar
Add value labels at the end of each bar
Rotate y-axis labels if needed
Apply plt.style.use('seaborn-v0_8')'''


# In[23]:


plt.style.use('seaborn-v0_8')


# In[24]:


cooking_counts = df['Cooking Method'].value_counts()


# In[29]:


fig, ax = plt.subplots(figsize=(10, 8))
colors = cm.tab20.colors[:len(cooking_counts)]  # Use distinct colors
bars = ax.barh(cooking_counts.index, cooking_counts.values, color=colors)
plt.title("Street Food by Cooking Method")
plt.xlabel("Number of Dishes")
plt.ylabel("Cooking Method")
ax.bar_label(bars, padding=3)  # Use built-in bar_label function
plt.gca().invert_yaxis()  # Highest count at the top


# In[ ]:


#I2. Grouped Bar Chart (15 points)
'''Create a grouped bar chart comparing vegetarian vs non-vegetarian dishes for the top 5 countries:
X-axis: Countries
Y-axis: Count of dishes
Two bars per country (Vegetarian: Yes/No)
Add legend and grid
Use contrasting colors'''


# In[30]:


top_countries = df['Country'].value_counts().head(5).index


# In[31]:


# Filter only rows from top countries
df_top = df[df['Country'].isin(top_countries)]


# In[33]:


grouped = df_top.groupby(['Country', 'Vegetarian']).size().unstack(fill_value=0)
grouped = grouped[['Yes', 'No']] if 'Yes' in grouped.columns and 'No' in grouped.columns else grouped


# In[34]:


plt.style.use('seaborn-v0_8')
x = np.arange(len(grouped.index))  # Country indices
width = 0.35  # Bar width


# In[37]:


fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, grouped['Yes'], width, label='Vegetarian', color='green')
bars2 = ax.bar(x + width/2, grouped['No'], width, label='Non-Vegetarian', color='red')


# In[38]:


ax.set_xlabel('Country')
ax.set_ylabel('Number of Dishes')
ax.set_title('Vegetarian vs Non-Vegetarian Dishes by Country')
ax.set_xticks(x)
ax.set_xticklabels(grouped.index)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)


# In[39]:


ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)


# In[40]:


plt.tight_layout()
plt.show()


# In[ ]:




