# IMPORT PACKAGES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from termcolor import colored

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression  # OLS algorithm
from sklearn.linear_model import Ridge  # Ridge algorithm
from sklearn.linear_model import Lasso  # Lasso algorithm
from sklearn.linear_model import BayesianRidge  # Bayesian algorithm
from sklearn.linear_model import ElasticNet  # ElasticNet algorithm

from sklearn.metrics import explained_variance_score as evs  # evaluation metric
from sklearn.metrics import r2_score as r2  # evaluation metric

# GET DATA AND FILTER COLUMNS
df = pd.read_excel(
    "C:/Users/frogb/Desktop/projects/data-science/homes/db/homes.xlsx")
fh = df[["Price", "Bathrooms", "Bedrooms", "Living Area",
         "Lot Size", "Home Type", "City", "Zip"]]

# DATA DISCOVERY

# fh.plot.scatter(x="Bathrooms",y="Price")
display(fh["Price"].min())
display(fh["Price"].max())
display(fh["Price"].mean())

display(fh["Bedrooms"].mean())
display(fh["Bathrooms"].mean())
display(fh["Living Area"].mean())
