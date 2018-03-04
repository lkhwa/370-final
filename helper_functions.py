import numpy as np
import pandas as pd
import seaborn as sns # for visualiation
import matplotlib.pyplot as plt # plotting
import matplotlib

business_data = pd.read_csv('./data/yelp_business.csv')
tip_data = pd.read_csv('./data/yelp_tip.csv')

def filter_category():
    new_df = business_data[['business_id', 'city', 'state', 'stars', 'categories']]
    return new_df[new_df['categories'].str.contains('Cafes|Coffee & Tea')]

def merge_id():
    df = filter_category()
    return pd.merge(df, tip_data[['business_id','text']], on='business_id')