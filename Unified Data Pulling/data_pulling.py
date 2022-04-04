#!/usr/bin/env python
# coding: utf-8

# In[133]:


import time
import json
import requests
import glob
import matplotlib.pyplot as plt
from datetime import datetime


# In[184]:


import numpy as np


# In[84]:


# In[85]:


#its bad practice to place your bearer token directly into the script (this is just done for illustration purposes)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAE%2FxYwEAAAAAD4yHOet7t9Teo7FvtNEHZgcrKd4%3DgZfLjnLxwc7QEfCOiPkdPEjjbPtOyBSWWQ4ccgm2o831mXxAfI"

#define search twitter function
def search_twitter(query, tweet_fields, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token), "tweet_mode":"extended"}

    url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}".format(
        query, tweet_fields
    )
    response = requests.request("GET", url, headers=headers)

    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# In[86]:


def get_tweets(key_words, start, end, debug=True, do_sleep=False):

    query_o = "-is%3Aretweet%20" + key_words + "&max_results=100&start_time=" + start + "&end_time=" + end

    fields = ['attachments',
              'author_id',
              'context_annotations',
              'conversation_id',
              'created_at',
              'entities',
              'geo',
              'id',
              'in_reply_to_user_id',
              'lang',
              'possibly_sensitive',
              'public_metrics',
              'referenced_tweets',
              'source',
              'text',
              'withheld']

    tweet_fields = 'tweet.fields=' + ''.join([word + ',' for word in fields])[:-1]
    
    responses = []
    next_token = ""
    
    json_response = search_twitter(query=query_o, tweet_fields=tweet_fields, bearer_token=BEARER_TOKEN)
    responses.append(json_response)
    
    if debug:
        print(json_response['meta']['result_count'])
    
    if do_sleep: time.sleep(1)
    
    while 'next_token' in json_response['meta']:
        
        next_token = json_response['meta']['next_token']
        query = query_o + "&next_token=" + next_token
        
        json_response = search_twitter(query=query, tweet_fields=tweet_fields, bearer_token=BEARER_TOKEN)
        responses.append(json_response)
        
        if debug:
            print(json_response['meta']['result_count'])
        
        if do_sleep: time.sleep(1)
    
    return responses


# In[87]:


def save_json(responses, file_path):
    with open(file_path) as outfile: json.dump(responses, outfile)


# In[111]:


def read_jsons(file_path):
    
    files = glob.glob(file_path)
    
    responses = []

    for file in files:
        f = open(file)
        data = json.load(f)
        responses += data['data']
    
    return responses


# In[120]:


def read_json(file_path):
    f = open(file_path)
    data = json.load(f)
    
    responses = []
    
    for item in data:
        responses+= item['data']
        
    return responses


# In[89]:


def response_to_data(responses):
    
    new_data = []
    for item in responses: new_data += responses['data']
    
    return new_data


# In[90]:


def search_users(ids, fields, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}

    url = "https://api.twitter.com/2/users?ids={}&user.fields={}&expansions=pinned_tweet_id".format(ids, fields)
    response = requests.request("GET", url, headers=headers)

    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def search_users_by_name(usernames, fields, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}

    url = "https://api.twitter.com/2/users/by/username/{}?expansions=pinned_tweet_id&user.fields={}".format(usernames, fields)
    response = requests.request("GET", url, headers=headers)

    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# In[91]:


def get_users(ids, search_func = search_users):
    
    user_data = []
    
    user_fields = ['public_metrics',
                   'created_at',
                   'description',
                   'entities',
                   'id',
                   'location',
                   'name',
                   'pinned_tweet_id',
                   'profile_image_url',
                   'protected',
                   'url',
                   'username',
                   'verified',
                   'withheld']

    fields = ''.join([word + ',' for word in user_fields])[:-1]
    
    i = -1
    for i in range(int(len(ids) / 100)):

        output_string = ''.join([item + ',' for item in ids[100 * i: 100 * (i + 1)]])[:-1]
        data = search_func(output_string, fields)
        user_data += data['data']
    
    
    output_string = ''.join([item + ',' for item in ids[100 * (i+1):]])[:-1]
    data = search_func(output_string, fields)
    user_data += data['data']
    
    return user_data


# In[92]:


def attach_users_to_tweets(tweets, users):
    
    for tweet in tweets: tweet['account_metrics'] = None
        
    ids = [tweet['author_id'] for tweet in tweets]
    
    for user in users:
        tweets[ids.index(user['id'])]['account_metrics'] = user
    
    return tweets


# In[93]:


def remove_none_ids(tweets):
    
    i = 0
    while i < len(tweets):
        if tweets[i]['account_metrics'] == None:
            tweets.pop(i)
        else:
            i += 1
    
    return tweets


# In[94]:


def remove_zero_followers(tweets):
    
    i = 0
    while i < len(tweets):
        if tweets[i]['account_metrics']['public_metrics']['followers_count'] == 0:
            tweets.pop(i)
        else:
            i += 1
    
    return tweets


# In[95]:


def date_score(metric):
    return datetime.strptime(metric['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()


# In[96]:


def total_engagement(metric):
    return sum(metric['public_metrics'][key] for key in metric['public_metrics'])


# In[97]:


def followers(metric):
    return metric['account_metrics']['public_metrics']['followers_count']


# In[98]:


def normalized_engagement(metric):
    return total_engagement(metric) / followers(metric)


# In[204]:


def normalized_engagement_smoothed(metric):
    return (total_engagement(metric) + 1)  / followers(metric)


# In[137]:


def get_len(score_arr):
    return len(score_arr)

# In[200]:


def binify(responses, num_bins, date_search_value, functions, score_function, debug=False):
    
    search_position = datetime.strptime(date_search_value, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
    
    times_and_metrics = []
    
    date_position = 0

    for tweet in responses:
        times_and_metrics.append([date_score(tweet)])
        
        for function in functions: times_and_metrics[-1].append(function(tweet))
        
        
        if date_score(tweet) < search_position: date_position += 1
            
    start = min([val[0] for val in times_and_metrics])
    end = max([val[0] for val in times_and_metrics])
    
    shift = (end - start) / num_bins
    
    x_pos = [start + (i + 1) * shift for i in range(num_bins)]
    
    y_pos = []
    
    max_range = [0, 0]
    
    search_pos = 0
    score = 0
    score_arr = []
    
    for time in x_pos:
        
        score = 0
        score_arr = []
        
        while times_and_metrics[search_pos][0] < time:

            score_arr.append(times_and_metrics[search_pos][1:])

            search_pos += 1
        
        y_pos.append(score_function(score_arr))
        if debug:
            print(score_arr)
            print(score_function(score_arr))
            print('----------')
        
    x_pos = [datetime.fromtimestamp(time_elem) for time_elem in x_pos]
        
    return x_pos, y_pos, date_position / len(responses)
        


def metric_hist(metric_function, user_data, bins=200):
    plt.figure(figsize=[16,9])
    plt.hist([metric_function(user) for user in user_data], bins = bins)
    plt.show()




