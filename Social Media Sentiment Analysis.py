#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[2]:


tweet = pd.read_csv("Tweets.csv")
len(tweet)


# In[3]:


tweet.head()


# In[4]:


tweet.describe()


# In[5]:


tweet.info()


# In[7]:


def deal_missing_values(X_full):
    #drop col where data is very less
    X_full = X_full.drop('airline_sentiment_gold', axis=1)
    X_full = X_full.drop('negativereason_gold', axis=1)
    X_full = X_full.drop('tweet_coord', axis=1)
    # replace null values with mean
    X_full['negativereason_confidence'] = X_full['negativereason_confidence'].fillna(X_full['negativereason_confidence'].mean())
    return X_full
    
tweet = deal_missing_values(tweet)
tweet.info()


# In[8]:


tweet.hist(bins = 30, figsize = (8,5))
plt.show()


# In[9]:


(tweet['airline'].unique())


# In[10]:


(tweet['negativereason'].unique())


# In[11]:


tweet.tail()


# In[19]:


X = tweet.drop('airline_sentiment', axis = 1)
y = tweet['airline_sentiment']


# In[20]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

ct = make_column_transformer(
    (MinMaxScaler(), ["tweet_id"]),
    (OneHotEncoder(handle_unknown="ignore"), ["airline", "retweet_count"])
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct.fit(X_train)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)


# In[22]:


lr_model = LogisticRegression(max_iter = 1000)
lr_model.fit(X_train_normal, y_train)
tree_model = SVC()
tree_model.fit(X_train_normal, y_train)


# In[23]:


y_pred = lr_model.predict(X_test_normal)
accuracy = accuracy_score(y_test, y_pred)
y_pred_tree = tree_model.predict(X_test_normal)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Accuracy: ", accuracy_tree)

