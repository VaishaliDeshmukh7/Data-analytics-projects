#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[32]:


df=pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\CDAC Big Data Anyaltics\Final_Project_CDAC\Stock_Data_ProjectTeam10\aligned_stock_data.csv")
df.head()


# In[33]:


df.shape


# In[34]:


df.info()


# In[35]:


df.isnull().sum()


# In[36]:


df.duplicated()


# In[37]:


df.duplicated().sum()


# In[38]:


df['Date']=pd.to_datetime(df['Date'],format="%d-%m-%Y")
df.head()


# In[39]:


df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
df.head()


# In[40]:


import matplotlib.pyplot as plt

# 1️⃣  Make sure the DataFrame is sorted by symbol and date
df = df.sort_values(['Symbol', 'Date'])

# 2️⃣  Create one figure and plot each symbol's close-price line
plt.figure(figsize=(14, 5))

for sym, sub in df.groupby('Symbol'):
    plt.plot(
        sub['Date'],
        sub['Close'],
        label=sym          # legend entry = symbol name
    )

plt.title("Stock Closing Price Over Time (Symbol-wise)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(title='Symbol', ncol=3, frameon=False)  # auto-builds legend
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[10]:


df.tail()


# # Goal : Predict the Next Day's Close Price

# # Now build more useful features for the model
# # 1. Intraday Volatility(High_Value)

# In[11]:


'''
Meaning:
The range between the highest and lowest price on that day.

Why it's useful:
Shows how much the stock fluctuated in a day.

High volatility might mean uncertainty or market reaction.

'''


# In[12]:


df['High_Low'] = df['High'] - df['Low']
df.head()


# # 2. Daily Return (Momentum)

# In[13]:


'''
Meaning:
The percentage change in close price compared to the previous day 
(for each stock separately).

Tells us whether the stock was going up or down recently.

Helps capture momentum — 
models can learn patterns like "price tends to rise after 3 up days."

* What does df.groupby('Symbol')['Close'].pct_change() do?
groupby('Symbol')

Splits the entire table into separate mini‑tables, one for each stock symbol (AAPL, MSFT, etc.).

This ensures Apple’s returns are calculated only from Apple’s prices, Microsoft’s from Microsoft’s, and so on.

* ['Close']

Focuses on each group’s Close‑price column—the last traded price of the day.

* For every row inside each group, it computes the percentage change from the previous row.
* Why add Daily_Return?
It captures momentum — whether the stock is rising or falling lately.

Machine‑learning models often find patterns like “after several positive returns, price tends to keep climbing” or the opposite for reversals.

Because we calculate it per symbol, each stock’s behaviour is learned independently and accurately. 

In simple words:
On 26th June, AAPL's stock closed 4.76% higher than it did on the previous day (25th June).
Why is this value important?
It shows how fast the stock is moving.

A positive return means the stock is going up, and a negative return means it's going down.

Models use this to detect momentum patterns like:

"If daily return has been positive for 3 days, price may keep going up."

"If daily return is very negative, maybe a reversal is coming."





'''


# In[14]:


df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
df.head()


# #  Moving Averages (Smoothing Trend)

# In[15]:


'''
MA5     = Average of last 5 close prices
MA10    = Average of last 10 close prices
MA20    = Average of last 20 close prices


Why do we add these MAs?
To give the model a sense of trend:

If Close > MA5, it’s moving up fast.

If MA5 > MA10 > MA20, a strong uptrend may be forming.

Used heavily in technical analysis.


Why do we compare Close vs MA5, MA5 vs MA10 vs MA20?
Because it helps us answer:

“Is the stock price going UP or DOWN lately?”

Case 1: If Close > MA5
Today's closing price is higher than the average of the last 5 days.

That means the stock is rising quickly right now.

The price is above its recent average → it’s moving up fast.


Case 2: If MA5 > MA10 > MA20
This means:

The short-term average (last 5 days) is higher than...

the medium-term average (last 10 days), which is higher than...

the long-term average (last 20 days)

This tells us:

The price has been steadily rising over time — the stock is in a strong uptrend.

Simple meaning:

The stock has been climbing smoothly, step by step.


Condition	            What it means       	      What model learns
Close > MA5	            Fast recent rise	          Price momentum
MA5 > MA10 > MA20	    Long-term steady uptrend	  Strong bullish trend
Close < MA5, MA5 < MA10 < MA20	Falling stock	      Downtrend pattern '''


# In[16]:


for win in [5, 10, 20]:
    df[f"MA{win}"] = df.groupby('Symbol')['Close'].transform(
                        lambda x: x.rolling(win).mean())


# In[17]:


df.head()


# # Next Close variable

# In[18]:


'''in simple word copying each stock’s next‑day closing price into the current row, then throwing away rows that can’t have a future price, so the model has a clear 
label (Next_Close) to predict from today’s data.'''


# In[19]:


for win in [5, 10, 20]:
    df[f"MA{win}"] = df.groupby('Symbol')['Close'].transform(
                        lambda x: x.rolling(win).mean())

df['Next_Close'] = df.groupby('Symbol')['Close'].shift(-1)
df = df.dropna().reset_index(drop=True)
df.head()


# In[20]:


df.shape


# # ONE‑HOT ENCODE THE SYMBOL COLUMN

# In[21]:


df = pd.get_dummies(df, columns=['Symbol'], prefix='Sym')
df.head()


# # DEFINE FEATURES (X) AND TARGET (y)

# In[22]:


# ─── 1.  List the numeric feature columns ─────────────────────────────────
num_feats = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'High_Low', 'Daily_Return', 'MA5', 'MA10', 'MA20'
]

# ─── 2.  Build a sub‑DataFrame that also includes the target ─────────────
corr_df = df[num_feats + ['Next_Close']]

# ─── 3.  Compute correlations column‑wise against Next_Close ─────────────
correlations = (
    corr_df.corr()['Next_Close']     # full correlation column
          .drop('Next_Close')        # drop self‑correlation (1.0)
          .sort_values(ascending=False)
)

# ─── 4.  Display nicely ──────────────────────────────────────────────────
print("Correlation with Next_Close:")
print(correlations)


# In[23]:


num_feats   = ['Open', 'High', 'Low', 'Close',
               'High_Low', 'MA5', 'MA10', 'MA20']
sym_feats   = [col for col in df.columns if col.startswith('Sym_')]
feature_cols = num_feats + sym_feats

X = df[feature_cols]
y = df['Next_Close']


# # TIME‑BASED TRAIN / TEST SPLIT (80 % , 20 %)
# # Using the global date order keeps the future hidden from the past.

# In[24]:


split_idx  = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# # TRAIN RANDOM FOREST ON ALL STOCKS

# In[25]:


# ──────────────────────────────────────────────────────────────
# Fast Random‑Forest tuning + final refit in one go
# Assumes: X_train, y_train, X_test, y_test already defined
# ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from scipy.stats import randint
import numpy as np

# 1) Base model: allow trees to parallelise, leave outer search single‑core
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

# 2) Compact hyper‑parameter space
param_dist = {
    'n_estimators'     : randint(150, 400),     # smaller range → faster
    'max_depth'        : [None, 10, 20],
    'min_samples_split': randint(2, 8),
    'min_samples_leaf' : randint(1, 6),
    'max_features'     : ['sqrt', 'log2', 0.5],
    'bootstrap'        : [True]
}

# 3) Time‑series‑aware CV
tscv = TimeSeriesSplit(n_splits=3)              # only 3 folds

# 4) Randomised search (20 combos × 3 folds = 60 fits)
rand_search = RandomizedSearchCV(
    estimator   = rf_base,
    param_distributions = param_dist,
    n_iter      = 20,
    scoring     = make_scorer(mean_absolute_error, greater_is_better=False),
    cv          = tscv,
    n_jobs      = 1,            # outer loop sequential (inner trees already parallel)
    random_state= 42,
    verbose     = 2
)

# 5) Fit search
rand_search.fit(X_train, y_train)
best_params = rand_search.best_params_
print("Best params (fast search):\n", best_params)

# 6) Refit best model with more trees for stability
best_params['n_estimators'] = 500               # beef up tree count
final_rf = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
final_rf.fit(X_train, y_train)

# 7) Evaluate on hold‑out test set
y_pred = final_rf.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nFinal model performance:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.4f}")


# In[26]:


y_naive  = X_test['Close']          # predict tomorrow = today
r2_naive = r2_score(y_test, y_naive)
print("Naïve R² :", r2_naive)


# In[ ]:




