from collections import Counter
from nltk import ngrams
# TODO
# from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from string import ascii_lowercase, ascii_uppercase
import nltk




#pre proccesing text
#remove/return links from text
#remove emojis
#remove/count @s  (usernames)
#remove/count #s (hashtags)
#elipsies??
#parsing all textual symbols 

# https://stackoverflow.com/questions/10677020/real-word-count-in-nltk
def number_of_words(text):
# TODO
#     regexptokenizer = RegexpTokenizer(r'\w+')
#     words = regexptokenizer.tokenize(text)
    words = word_tokenize(text)
    return len(words)

import advertools as adv
def get_emoji(orig_list):
    emoji_dict = adv.extract_emoji(orig_list)
    return emoji_dict

def number_of_character_1_grams(text):
    characters = [c for c in text]
    onegrams = ngrams(characters, 1)
    return len([gram for gram in onegrams])


def number_of_character_2_grams(text):
    if len(text) == 0:
        return []
    characters = [c for c in text]
    twograms = ngrams(characters, 2)
    return len([gram for gram in twograms])

def number_of_character_3_grams(text):
    if len(text) <= 1:
        return 0
    characters = [c for c in text]
    threegrams = ngrams(characters, 3)
    return len([gram for gram in threegrams])


# https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
def clindex(text):
    text_lower = text.lower()
    number_of_letters = 0
    for character in text_lower:
        if character in ascii_lowercase:
            number_of_letters += 1
    number_of_sentences = len(sent_tokenize(text))
    n_of_words = number_of_words(text)
    l = 0
    s = 0
    # TODO should l and s be 0?
    if n_of_words == 0:
        pass
    else:
        # l = Letters ÷ Words × 100
        l = number_of_letters / n_of_words * 100
        # s = Sentences ÷ Words × 100
        s = number_of_sentences / n_of_words * 100
    return 0.0588 * l - 0.296 * s - 15.8

def formality_measure(text):
    tokenized_text = nltk.word_tokenize(text.lower())
    t = nltk.Text(tokenized_text)
    pos_tags = nltk.pos_tag(t)
    counts = Counter(tag for word,tag in pos_tags)
    return (counts['NN'] + counts['NNP'] + counts['NNS'] + counts['JJ'] + counts['JJR'] + counts['JJS'] + counts['IN'] + counts['DT'] + counts['PDT'] + counts['WDT'] - counts['PRP'] - counts['PRP$'] - counts['WP'] - counts['WP$'] - counts['VB'] - counts['VBD'] - counts['VBG'] - counts['VBN'] - counts['VBP'] - counts['VBZ'] - counts['RB'] - counts['RBR'] - counts['RBS'] - counts['WRB'] - counts['UH'] + 100) / 2


def is_exclamation_question_mark_present(text):
    return 0 if '!' not in text and '?' not in text else 1


def lix(text):
    # TODO should we return 0?
    if len(sent_tokenize(text)) == 0:
        return 0
    return number_of_words(text) / len(sent_tokenize(text))


def number_of_uppercase_words(text):
    words = word_tokenize(text)
    n_of_uppercase_words = 0
    for word in words:
        if word[0] in ascii_uppercase:
            n_of_uppercase_words += 1
    return n_of_uppercase_words


def rix(text):
    lw = 0
    words = word_tokenize(text)
    for word in words:
        if len(word) >= 7:
            lw += 1
    # TODO should we return 0?
    if len(sent_tokenize(text)) == 0:
        return 0
    return lw / len(sent_tokenize(text))

def number_of_word_1_grams(text):
    onegrams = ngrams(word_tokenize(text), 1)
    return len([gram for gram in onegrams])

import pandas as pd

from ttp import ttp
p = ttp.Parser()
def get_url(text):
    result = p.parse(text)

    return str(result.urls)

df = pd.read_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/data2.csv")
df = df.dropna()
df_urls = pd.DataFrame()
df_urls["url"] = None
df_urls["label"] = None
for i in df.index:
    df_urls.at[i, 'url'] = get_url(df.loc[i]['text'])
    df_urls.at[i, 'label'] = df.loc[i]['clickbait']



df_urls.to_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/df_urls.csv")

# print(len(df[df["clickbait"]=="no-clickbait"]))
# print(len(df[df["clickbait"] == "clickbait"]))
#
# df = df.iloc[: , 6:]
# from ttp import ttp
# p = ttp.Parser()
# def get_text(text):
#
#     text_fin = ""
#
#     result = p.parse(text)
#
#
#     arr = text.split(" ")
#
#     for elem in arr:
#         if ((elem not in str(result.users)) and (elem not in str(result.tags)) and (elem not in str(result.urls))):
#             text_fin = text_fin  + elem + " "
#
#     return text_fin[:-1]
#
#
# df['number_of_character_1_grams'] = None
# df['number_of_character_2_grams'] = None
# df['number_of_character_3_grams'] = None
# df['clindex'] = None
# df['formality_measure'] = None
# df['is_exclamation_question_mark_present'] = None
# df['lix'] = None
# df['number_of_uppercase_words'] = None
# df['number_of_words'] = None
# df['rix'] = None
# df['number_of_word_1_grams'] = None
# for i in df.index:
#     df.at[i, 'number_of_character_1_grams'] = number_of_character_1_grams(get_text(df.loc[i]['text']))
#     df.at[i, 'number_of_character_2_grams'] = number_of_character_2_grams(get_text(df.loc[i]['text']))
#     df.at[i, 'number_of_character_3_grams'] = number_of_character_3_grams(get_text(df.loc[i]['text']))
#     df.at[i, 'clindex'] = clindex(get_text(df.loc[i]['text']))
#     df.at[i, 'formality_measure'] = formality_measure(get_text(df.loc[i]['text']))
#     df.at[i, 'is_exclamation_question_mark_present'] = is_exclamation_question_mark_present(get_text(df.loc[i]['text']))
#     df.at[i, 'lix'] = lix(get_text(df.loc[i]['text']))
#     df.at[i, 'number_of_uppercase_words'] = number_of_uppercase_words(get_text(df.loc[i]['text']))
#     df.at[i, 'number_of_words'] = number_of_words(get_text(df.loc[i]['text']))
#     df.at[i, 'rix'] = rix(get_text(df.loc[i]['text']))
#     df.at[i, 'number_of_word_1_grams'] = number_of_word_1_grams(get_text(df.loc[i]['text']))
#
# print(df)
# df.to_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/features.csv")