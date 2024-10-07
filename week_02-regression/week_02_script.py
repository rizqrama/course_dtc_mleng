# %%
# Import libraries
import pandas as pd
import numpy as np

# %%
# Download the dataset
car_data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
!wget -P week_02-regression/data/ $car_data

car_raw = pd.read_csv('week_02-regression/data/data.csv')

car_raw.head()

# %% Data preparation
## tidy the dataset
car_raw.columns = car_raw.columns.str.lower().str.replace(' ','_')

car_raw.head()