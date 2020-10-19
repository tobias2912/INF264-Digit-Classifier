
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import holidays
'''
Methods for feature engineering.
either adds a column to a dataframe obj or removes columns
'''


def remove_columns(df):
    '''remove all columns not used as features'''
    del df['datetime']
    del df['Måned']
    del df['Dag']
    del df['År']

