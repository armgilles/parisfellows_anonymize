# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:24:13 2016

@author: babou
"""

import pandas as pd
import glob
import nltk
import random

from nltk.corpus import stopwords
from docx import Document
from sklearn.preprocessing import LabelEncoder


#target_list = ['X\xe2\x80\xa6', 'Y\xe2\x80\xa6', 'Z\xe2\x80\xa6', 'X']

stopword_fr = [word for word in stopwords.words('french')]

target_dict = {u'X\xe2\x80\xa6' : 'X', u'X' : 'X', u'X..' : 'X', u'X.' : 'X', 
               u'X\u2026' : 'X', u'M.X' : 'X', u'M.X\u2026' : 'X',
               u'Y\xe2\x80\xa6' : 'Y', u'Y' : 'Y', u'Y..' : 'Y', u'Y.' : 'Y', 
               u'Y\u2026' : 'Y', u'M.Y' : 'Y', u'M.Y\u2026' : 'Y',
               u'Z\xe2\x80\xa6': 'Z', u'Z' : 'Z', u'Z..' : 'Z', u'Z.' : 'Z', 
               u'Z\u2026' : 'Z', u'M.Z' : 'Z', u'M.Z\u2026' : 'Z'}

#punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

#################################################
###                 FUNCTIONS                 ###
#################################################


#def get_target_name(sentence):
#    for i in range(len(sentence.split(' '))):
#        w = sentence.split(' ')[i].replace(',', '').replace('.', '')
#        if w in target_list:
#            return target_dict.keys.index(w)


def get_target(sentence):
    for i in range(len(sentence.split(' '))):
        w = sentence.split(' ')[i].replace(',', '').replace('.', '')
        if w in target_dict.keys():
            return 1
            
#def remove_ponctuation(sentence):
#    sentence_with_no_punct = ""
#    for char in sentence:
#       if char not in punctuations:
#           sentence_with_no_punct = sentence_with_no_punct + char
#    return sentence_with_no_punct
    


#################################################
###                 GENERAL                   ###
#################################################

path_files = glob.glob('data/jugements CRC LÇgifrance/*')
doc = Document(path_files[0])

documents_df = pd.DataFrame()
document_temp = pd.DataFrame()


for path_file in path_files:
    print "Loading file : " + path_file.split('/')[-1]
    if path_file.split('.')[-1] == "docx":
        doc = Document(path_file)
        paragraphs = doc.paragraphs
        my_document = []
        for paragraph in paragraphs:
            my_document.append({'doc_name' : path_file.split('/')[-1],
                                'paragraph' : paragraph.text})
                                
        document_temp = pd.DataFrame(my_document)
        documents_df = pd.concat([documents_df, document_temp])
    else:
        print "Error - Not a docx file : " + path_file.split('/')[-1]

#documents_df['paragraph_no_punct'] = documents_df['paragraph'].apply(lambda x: remove_ponctuation(x))
documents_df.reset_index(inplace=True)
documents_df['is_target'] = documents_df['paragraph'].apply(lambda x: get_target(x))
documents_df['is_target'].fillna(0, inplace=True)


print documents_df.is_target.value_counts()
print documents_df.groupby('doc_name')['is_target'].sum()


# Old 
#    tags = tagger.TagText(row.['paragraph'])
#    tags2 = treetaggerwrapper.make_tags(tags)
#    for w in tags2:
#        word_list.append({'word' : w[0],
#                          'type' : w[1],
#                            'lemma' : w[2]})

print "-"*54
print " PREPROCESSING..."

word_list = []
context_word =[]
for idx , row in documents_df.T.iteritems():
    tokenized = nltk.word_tokenize(row['paragraph'])
    word_list.extend(tokenized)
    # Some context paragraph / doc
    for len_token in tokenized:
        context_word.append({'doc_name' : row['doc_name'],
                             'paragraph_nb' : row['index']})

# My bad of word
word_df = pd.DataFrame(word_list, columns=['word'])

# My context 
context_df = pd.DataFrame(context_word)

# Merging context & bad of word
word_df = pd.concat([word_df, context_df], axis=1)

# Reading firstnames file
firstname_df = pd.read_csv('data/prenom_clean.csv', encoding='utf-8', 
                           header=None, names = ["firstname"])
# Use Majuscule in first carac                           
firstname_df['firstname'] = firstname_df['firstname'].apply(lambda x: x.title())                           
firstname_list = firstname_df.firstname.tolist()


word_df['is_target'] = word_df['word'].apply(lambda x: 1 if x in target_dict.keys() else 0)
word_df['is_stopword'] = word_df['word'].apply(lambda x: 1 if x.lower() in stopword_fr else 0)
word_df['is_first_char_upper'] = word_df['word'].apply(lambda x: 1 if x[0].isupper() else 0)
word_df['is_upper'] = word_df['word'].apply(lambda x: 1 if x.isupper() else 0)
word_df['is_firstname'] = word_df['word'].apply(lambda x: 1 if x.lower() in firstname_list  else 0)

# to have granularite
word_df['temp_count'] = 1 

# Cumulative sum of word by paragraph
word_df['paragraph_cum_word' ] = word_df.groupby(['doc_name', 'paragraph_nb'])['temp_count'].cumsum()  

# Cumulative sum of word by senstence end by ";" or "."
word_df['temp_count'] = 1
## Create a bool a each end of sentence
word_df["end_point"] = word_df.word.apply(lambda x: 1 if x in [";", "."] else 0)

word_df['end_point_cum' ] = word_df.groupby(['doc_name'])['end_point'].cumsum()
word_df['end_point_cum_word' ] = word_df.groupby(['doc_name', 'end_point_cum'])['temp_count'].cumsum()
#word_df = word_df.drop(['temp_count', 'end_point', 'end_point_cum'], axis=1)

# Cumulative sum of word by senstence end by ","
## Create a bool a each end of sentence
word_df["end_comma"] = word_df.word.apply(lambda x: 1 if x in [","] else 0)
## If end of sentence "." & ";" then end of comma to
word_df.loc[word_df['end_point'] == 1, "end_comma"] = 1

word_df['end_comma_cum' ] = word_df.groupby(['doc_name'])['end_comma'].cumsum()
word_df['end_comma_cum_word' ] = word_df.groupby(['doc_name', 'end_comma_cum'])['temp_count'].cumsum()

# Del temp preprocessing features
word_df = word_df.drop(['temp_count', 'end_comma', 'end_comma_cum',
                       'end_point', 'end_point_cum'], axis=1)


# Insert random first name in place of 'X', Mr 'X...' ...
# /!\ Delete when true data
def get_random_firstname(x):
    random_idx = random.randint(0, 2046) # len de firstname_list
    return firstname_list[random_idx]
    
# To delete after
word_df.loc[word_df['is_target'] ==1, 'word'] = word_df['word'].apply(lambda x: get_random_firstname(x))
word_df.loc[word_df['is_target'] ==1, 'is_firstname'] = random.randint(0, 1)    #1 is To strong rule so random
word_df['len_word'] = word_df['word'].apply(lambda x: len(x))  # len

# Label encoding word
lbl = LabelEncoder()
word_df['word_encoded'] = lbl.fit_transform(list(word_df['word'].values))
#word_df['word_encoded'] = word_df['word_encoded'].astype('str')

# Shift words encoded
## One word before 
word_df['word_encoded_shift_1b'] = word_df.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(1))
#word_df['word_encoded_shift_1b'] = word_df['word_encoded_shift_1b'].astype('int')
## Two words before 
word_df['word_encoded_shift_2b'] = word_df.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(2))
#word_df['word_encoded_shift_2b'] = word_df['word_encoded_shift_2b'].astype('int')
## One word after
word_df['word_encoded_shift_1a'] = word_df.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(-1))
#word_df['word_encoded_shift_1a'] = word_df['word_encoded_shift_1a'].astype('int')
## Two words after 
word_df['word_encoded_shift_2a'] = word_df.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(-2))
#word_df['word_encoded_shift_2a'] = word_df['word_encoded_shift_2a'].astype('int')

# Fillna Nan word shift
word_df = word_df.fillna(-1)

print " EXPORT..."
word_df.to_csv('data/data.csv', encoding='utf-8', index=False)


# OLD 
#docText = '\n\n'.join([
#    paragraph.text.encode('utf-8') for paragraph in paragraphs
#])


# tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGDIR='/Users/babou/Desktop/NLP',
#  TAGINENC='utf-8',TAGOUTENC='utf-8')