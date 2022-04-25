import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import sys



# encoder = preprocessing.LabelEncoder()
# loaded_model = joblib.load('/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav')
#text = 'Wow is this good. Every would-be startup founder should read this. Everyone who cares about the future, because this is what the way forward sounds like, not the whining and cynicism that fills Twitter and the media.'
# valid_y = encoder.fit_transform([text])
# print(valid_y)

#1 = no clickbait, 0 = clickbait

# def get_prediction(text, filepath):
#
#
#     # my file path = '/Users/shellyschwartz/PycharmProjects/diplomacy_lab/finalized_model.sav'
#     #./finalized_model.sav
#     loaded_model = joblib.load('./finalized_model.sav')
#     tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
#     df = pd.DataFrame("data2.csv")
#     df = df.dropna()
#     tfidf_vect.fit(df['text'])
#     valid_tfidf = tfidf_vect.transform([text])
#     predictions = loaded_model.predict(valid_tfidf)
#
#     # for pred in predictions:
#     #     if(pred == 0):
#     #         fin.append("clickbait")
#     #     elif(pred == 1):
#     #         fin.append("not clickbait")
#
#     print(predictions)


loaded_model = joblib.load('./finalized_model.sav')
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
df = pd.read_csv('./data2.csv')
df = df.dropna()
tfidf_vect.fit(df['text'])
valid_tfidf = tfidf_vect.transform(sys.argv[0])
predictions = loaded_model.predict(valid_tfidf)

    # for pred in predictions:
    #     if(pred == 0):
    #         fin.append("clickbait")
    #     elif(pred == 1):
    #         fin.append("not clickbait")

print(predictions)