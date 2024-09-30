# %%
import pandas as pd
import numpy as np

# %%
# 1. pandas version
pd.__version__
# %%
# load data
laptop_raw = pd.read_csv('data/laptops.csv')
laptop_raw.head()

# %%
# 2. records count
len(laptop_raw)

# %%
# 3. count of laptop brands
laptop_raw['Brand'].nunique()
# %%
# 4. count of column that has missing values
laptop_raw.isnull().sum()
# %%
# 5. maximum Dell notebook's final price
max(laptop_raw[laptop_raw['Brand'] == 'Dell']['Final Price'])
# %%
# 6.1. median value of 'Screen'

laptop_raw['Screen'].median()

# %%
# 6.2. most frequent value of 'Screen'
laptop_raw['Screen'].mode()[0]
# %%
# 6.3. fill na values in screen with that most frequent value
laptop_raw['Screen'] = laptop_raw['Screen'].fillna(laptop_raw['Screen'].mode()[0])

# %%
# 6.4. check again
median(laptop_raw['Screen'])
# %%
# 7.1 - 7.3 select columns RAM, Screen, Storage for Innjoo laptops and convert to numpy array
X = laptop_raw[laptop_raw['Brand'] == 'Innjoo'][['RAM', 'Storage', 'Screen']].to_numpy()
X

# %%
# 7.4 - 7.5 matrix-matrix multiplication and creating inverse
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
XTX_inv

# %%
# 7.6 the array y
y = np.array([1100, 1300, 800, 900, 1000, 1100])
y

# %%
# 7.7 Multiply the inverse of XTX with the transpose of X, and then multiply by y
w = XTX_inv @ X.T @ y
w

# %%
# 7.8: Sum of all elements of w
w_sum = np.sum(w)
w_sum
