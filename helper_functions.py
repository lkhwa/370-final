import numpy as np
import pandas as pd
import seaborn as sns # for visualiation
import matplotlib.pyplot as plt # plotting
import matplotlib
import re 
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from os import path
from PIL import Image
from wordcloud import WordCloud

# Load data
business_data = pd.read_csv('./data/yelp_business.csv')
tip_data = pd.read_csv('./data/yelp_tip.csv')

def business_data_head():
    return business_data.head()

def tip_data_head():
    return tip_data.head()

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
def get_tips(list, df):
    pd.set_option('display.max_colwidth', -1)
    new = pd.DataFrame()
    for word in list:
        df1 = df[['business_id', 'text', 'stars']][df['text'].str.contains(word)]
        df1 = df1.groupby('stars').agg({'business_id': 'first', 'text': 'first'}).reset_index()
        df1['word'] = word
        new = new.append(df1, ignore_index=True)
    return new

# Returns a dataframe which contains the top 100 businesses in terms of number of tips received
# Dataframe also contains the text of all the tips for a business, rating,
# and the frequency of occurences of each word in tips (based on a predefined list of words)
def top_businesses_by_tip(df, word_list):
    pd.set_option('max_colwidth',40)
    df1 = df[['business_id', 'text','state', 'stars']]
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
    fig.suptitle("Stars Distribution for Cafes|Coffee & Tea Business Categories", fontsize=16)
    ax.set_ylabel('Number of Businesses', fontsize=18)
    ax.set_xlabel('Stars (Ratings)', fontsize=18)
    sns.barplot(x.index, x.values)
    
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    plt.show()

# Create a word cloud using the top 100 words in tips
def create_wordcloud(word_list):
    words = [x[0] for x in word_list]
    d = path.dirname("__file__")

    # read the mask image
    coffee_mask = np.array(Image.open(path.join(d, "mask.png")))

    # generate word cloud
    wc = WordCloud(background_color="white", max_words=200, mask=coffee_mask)
    wc.generate(str(words))

    # store to file
    wc.to_file(path.join(d, "coffee_wordcloud.png"))

    # show
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Return three barchart that shows the distribution of 5 words each of three sentiments
# in the Coffee & Tea business category
def words_freq_distribution(df):
    love = df['love'].sum()
    friendly = df['friendly'].sum()
    free = df['free'].sum()
    fast = df['fast'].sum()
    happy = df['happy'].sum()
    wait = df['wait'].sum()
    closed= df['closed'].sum()
    busy = df['busy'].sum()
    long = df['long'].sum()
    slow = df['slow'].sum()
    service = df['service'].sum()
    atmosphere = df['atmosphere'].sum()
    location = df['location'].sum()
    wifi = df['wifi'].sum()
    donuts = df['donuts'].sum()

    pos = {'words': ['love', 'friendly', 'free', 'fast', 'happy'], 'sum':[love, friendly, free, fast, happy]}
    net = {'words': ['wait', 'closed', 'busy', 'long', 'slow'], 'sum':[wait, closed, busy, long, slow]}
    neg = {'words': ['service', 'atmosphere', 'location', 'wifi', 'donuts'], 'sum':[service, atmosphere, location, wifi, donuts]}
    pos_df = pd.DataFrame(data=pos)
    net_df = pd.DataFrame(data=net)
    neg_df = pd.DataFrame(data=neg)

    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    sns.barplot("words", y="sum", data=pos_df, palette="Blues_d", ax=axes[0])
    sns.barplot("words", y="sum", data=net_df, palette="Blues_d", ax=axes[1])
    sns.barplot("words", y="sum", data=neg_df, palette="Blues_d", ax=axes[2])
    fig.suptitle("Words Distribution for Coffee & Tea Category", fontsize=16)
    axes[0].set_title('Positive Words')
    axes[1].set_title('Neutral Words')
    axes[2].set_title('Negative Words')
    plt.show()

# For each rating category, 
# create a barchart to show the distribution of each sentiment category (positive, negative, neutral)
def top_words_distribution_rating(df):
    temp = df.loc[:, 'rating':].groupby("rating").sum()
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 10), sharey=True)
    n = 0
    for i, row in temp.iterrows():
        max_value_pos = row[0:5].max()
        max_idx_pos = row[0:5].argmax()
        max_value_neg = row[5:10].max()
        max_idx_neg = row[5:10].argmax()
        max_value_neu = row[10:].max()
        max_idx_neu = row[10:].argmax()

        labels = [max_idx_pos, max_idx_neg, max_idx_neu]
        label_pos = np.arange(len(labels))
        data = [max_value_pos, max_value_neg, max_value_neu]

        axes[n].bar(label_pos, data)
        axes[n].set_xticks(label_pos)
        axes[n].set_xticklabels(labels)
        axes[n].set_title("Rating: " + str(i))
        n += 1
    
    plt.show()

