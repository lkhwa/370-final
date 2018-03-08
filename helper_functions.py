import numpy as np
import pandas as pd
import seaborn as sns # for visualiation
import matplotlib.pyplot as plt # plotting
import matplotlib
import re 
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob
from nltk import FreqDist

# Load data
business_data = pd.read_csv('./data/yelp_business.csv')
tip_data = pd.read_csv('./data/yelp_tip.csv')

# Select businesses under 'Cafes' and 'Coffee & Tea' categories
def filter_category():
    df = business_data[['business_id', 'city', 'state', 'stars', 'categories']]
    return df[df['categories'].str.contains('Cafes|Coffee & Tea')]

# Merge business ids (with text of tips) from tips dataframe
def merge_id():
    df = filter_category()
    return pd.merge(df, tip_data[['business_id','text']], on='business_id')

# Process string to remove punctuation, stopwords and numerical characters
def process_string(s):
    string = re.split("\r\n", s)
    result = ""
    for substring in string:
        result += substring + " "
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(result.lower())
    remove_stopwords = [word for word in word_tokens if word not in stop_words]
    return [word for word in remove_stopwords if word.isalpha()]

# Returns 100 most frequently used words across all tips
def top_hundred(df):
    string = (' '.join(df['text']))
    fdist = FreqDist(process_string(string))
    return fdist.most_common(100)

# This function is for tip analysis purpose
# For each word in the list, get an example of tip (which contains the word) for each level of star rating
def get_tips(list, data):
    pd.set_option('display.max_colwidth', -1)
    new = pd.DataFrame()
    for word in list:
        df1 = data[['business_id', 'text', 'stars']][data['text'].str.contains(word)]
        df1 = df1.groupby('stars').agg({'business_id': 'first', 'text': 'first'}).reset_index()
        df1['word'] = word
        new = new.append(df1, ignore_index=True)
    return new

# Returns a dataframe which contains the top 100 businesses in terms of number of tips received
# Dataframe also contains the text of all the tips for a business, rating,
# and the frequency of occurences of each word in tips (based on a predefined list of words)
def top_businesses_by_tip(data, word_list):
    pd.set_option('max_colwidth',40)
    df1 = data[['business_id', 'text','state', 'stars']]
    df1 = df1.groupby('business_id').agg({'state': len, 'text': lambda x: ' '.join(x), 'stars': 'mean'}).reset_index().sort_values(by=['state'], ascending=False)
    df1.rename(columns={'state': 'tip_count', 'stars': 'rating'}, inplace=True)
    
    # Count the number of times each word appears in all tips
    for word in word_list:
        df1[word] = df1['text'].apply(lambda text: text.count(word))
    return df1.head(n=100)

# Returns a barchart that shows the distribution of star ratings in the Coffee & Tea business category
def ratings_distribution(df):
    x = df['stars'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(13,8))
    fig.suptitle("Stars Distribution for Coffee & Tea Category", fontsize=16)
    ax.set_ylabel('Number of Businesses', fontsize=18)
    ax.set_xlabel('Stars (Ratings)', fontsize=18)
    sns.barplot(x.index, x.values)
    
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    plt.show()

