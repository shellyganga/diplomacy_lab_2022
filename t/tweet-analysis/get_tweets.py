import time
import requests
import json
import pandas as pd


# INSERT BEARER TOKEN HERE
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJUwYwEAAAAAE7TyF7ND4RPHwVD5YEsFtqm5wUI%3D2PAsqcrjWj8QDBg3Gn3NPspRmGLDLplj67kRAhyUvsUdNqNDUm"

#twitter search function
def search_twitter(tweet_id, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    # Change this url to get differnt things off of Twitter Id
    url = "https://api.twitter.com/2/tweets/{}?user.fields=public_metrics,created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,url,username,verified,withheld".format(tweet_id)
    response = requests.request("GET", url, headers=headers)

    #print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

all_responses = []
import time
import requests
import json

# INSERT BEARER TOKEN HERE
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJUwYwEAAAAAE7TyF7ND4RPHwVD5YEsFtqm5wUI%3D2PAsqcrjWj8QDBg3Gn3NPspRmGLDLplj67kRAhyUvsUdNqNDUm"

#twitter search function
def search_twitter(tweet_id, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    # Change this url to get differnt things off of Twitter Id
    url = "https://api.twitter.com/2/tweets/{}".format(tweet_id)
    response = requests.request("GET", url, headers=headers)

    #print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

data1 = pd.read_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/data2.csv")
data2 = pd.DataFrame()




# for index, row in data1.iterrows():
#     if index >= beg and index <= end:
#         json_response = search_twitter(tweet_id=int(row['607668877594497024']), bearer_token=BEARER_TOKEN)
#         print(json_response)
#         keys = list(json_response.keys())
#         if 'errors' not in keys:
#             text = ""
#             try:
#                 text = json_response['data']['text']
#             except:
#                 text = "Not a tweet"
#
#             row['text'] = text
#     data2 = data2.append(row, ignore_index=True)
#
#
# data2.to_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/data2.csv")
beg = 1601
end = 1801
index = 0
while index<len(data1):
    if index >= beg and index <= end:

        json_response = search_twitter(tweet_id=int(data1.loc[index, '607668877594497024']), bearer_token=BEARER_TOKEN)
        print(json_response)
        keys = list(json_response.keys())
        if 'errors' not in keys:
            text = ""
            try:
                text = json_response['data']['text']
            except:
                text = "Not a tweet"

            data1.loc[index, 'text'] = text
    data2 = data2.append(data1.loc[[index]], ignore_index=True)
    if index>=end:
        print(data2.iloc[beg:end])
        data2.to_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/data2.csv")
        time.sleep(60*15)
        beg = beg + 200
        end = end + 200
    index = index + 1



