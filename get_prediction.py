# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:21:42 2016

@author: babou
"""

import pandas as pd
from docx import Document
import pickle

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import xgboost as xgb


## Check type of word
#def type_extraction(x):
#    """
#    Return type of word
#    """
#    try:
#        blob = TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
#        return blob.tags[0][1]
#    except:
#         return 'PONCTUATION'


def reading_docx(path_file):
    """
    Read a docx
    Return a DataFrame with word by line
    """
    
    print "Loading file : " + path_file.split('/')[-1]
    if path_file.split('.')[-1] == "docx":
        doc = Document(path_file)
        paragraphs = doc.paragraphs
        my_document = []
        for paragraph in paragraphs:
            my_document.append({'doc_name' : path_file.split('/')[-1],
                                'paragraph' : paragraph.text})

        documents_df = pd.DataFrame(my_document)
        documents_df.reset_index(inplace=True)
        
        word_list = []
        context_word =[]
        for idx , row in documents_df.T.iteritems():
            tokenized = nltk.word_tokenize(row['paragraph'])
            word_list.extend(tokenized)
            # Some context paragraph / doc
            for len_token in tokenized:
                context_word.append({'doc_name' : row['doc_name'],
                                     'paragraph_nb' : row['index']})
                                     
        # My bag of word
        word_df = pd.DataFrame(word_list, columns=['word'])
        
        # My context
        context_df = pd.DataFrame(context_word)
        
        # Merging context & bad of word
        word_df = pd.concat([word_df, context_df], axis=1)


        return word_df
    else:
        print "Error - Not a docx file : " + path_file.split('/')[-1]
        
def process_data(data):
    """
    Process data to create feature engi
    Return a dataframe
    """
    # to have granularite
    data['temp_count'] = 1
    
    # Cumulative sum of word by paragraph
    data['paragraph_cum_word' ] = data.groupby(['doc_name', 'paragraph_nb'])['temp_count'].cumsum()
    
    # Cumulative sum of word by senstence end by ";" or "."
    data['temp_count'] = 1
    ## Create a bool a each end of sentence
    data["end_point"] = data.word.apply(lambda x: 1 if x in [";", "."] else 0)
    
    data['end_point_cum' ] = data.groupby(['doc_name'])['end_point'].cumsum()
    data['end_point_cum_word' ] = data.groupby(['doc_name', 'end_point_cum'])['temp_count'].cumsum()
    #data = data.drop(['temp_count', 'end_point', 'end_point_cum'], axis=1)
    
    # Cumulative sum of word by senstence end by ","
    ## Create a bool a each end of sentence
    data["end_comma"] = data.word.apply(lambda x: 1 if x in [","] else 0)
    ## If end of sentence "." & ";" then end of comma to
    data.loc[data['end_point'] == 1, "end_comma"] = 1
    
    data['end_comma_cum' ] = data.groupby(['doc_name'])['end_comma'].cumsum()
    data['end_comma_cum_word' ] = data.groupby(['doc_name', 'end_comma_cum'])['temp_count'].cumsum()
    
    # Del temp preprocessing features
    data = data.drop(['temp_count', 'end_comma', 'end_comma_cum',
                           'end_point', 'end_point_cum'], axis=1)
    
    
    
    data['is_stopword'] = data['word'].apply(lambda x: 1 if x.lower() in stopword_fr else 0)
    data['is_first_char_upper'] = data['word'].apply(lambda x: 1 if x[0].isupper() else 0)
    data['is_upper'] = data['word'].apply(lambda x: 1 if x.isupper() else 0)
    data['is_firstname'] = data['word'].apply(lambda x: 1 if x in firstname_list else 0)
    data['len_word'] = data['word'].apply(lambda x: len(x))  # len
    
    
    # Shift is_firstname
    data['is_firstname_1b'] = data.groupby(['doc_name'])['is_firstname'].apply(lambda x: x.shift(-1))
    data['is_firstname_2b'] = data.groupby(['doc_name'])['is_firstname'].apply(lambda x: x.shift(-2))
    
    data['is_firstname_1a'] = data.groupby(['doc_name'])['is_firstname'].apply(lambda x: x.shift(1))
    data['is_firstname_2a'] = data.groupby(['doc_name'])['is_firstname'].apply(lambda x: x.shift(2))
    
        
    #Check is there is a "Mr", "M" before...
    data['is_mister_word'] = 0
    data.loc[data['word'].isin(mister_list), 'is_mister_word'] = 1
    
    ## Label encoding word with our bag of word
    data['word_stem'] = data['word'].apply(lambda x: stemmer.stem(x)) # if you want to stem it
    data['word_encoded'] = data.word_stem.apply(lambda x: dico_word.get(x, -1))
    
    # Shift words encoded
    ## One word before
    data['word_encoded_shift_1b'] = data.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(-1))
    
    ## Two words before
    data['word_encoded_shift_2b'] = data.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(-2))
    
    ## One word after
    data['word_encoded_shift_1a'] = data.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(1))
    
    ## Two words after
    data['word_encoded_shift_2a'] = data.groupby(['doc_name'])['word_encoded'].apply(lambda x: x.shift(2))
    
    # Shift is_mister_word Features
    ## Beforef 
    data['is_mister_word_1b'] = data.groupby(['doc_name'])['is_mister_word'].apply(lambda x: x.shift(1))
    data['is_mister_word_2b'] = data.groupby(['doc_name'])['is_mister_word'].apply(lambda x: x.shift(2))
    ## After
    data['is_mister_word_1a'] = data.groupby(['doc_name'])['is_mister_word'].apply(lambda x: x.shift(-1))
    data['is_mister_word_2a'] = data.groupby(['doc_name'])['is_mister_word'].apply(lambda x: x.shift(-2))
    
    # Type of word
#    data['word_type'] = data['word'].apply(lambda x: type_extraction(x))
#    data['word_type'] = data['word_type'].apply(lambda x: dico_word_type_info.get(x, -1)) # To get the same encoding
#    
#    # Shift Word type
#    data['word_type_1b'] = data.groupby(['doc_name'])['word_type'].apply(lambda x: x.shift(1))
#    data['word_type_2b'] = data.groupby(['doc_name'])['word_type'].apply(lambda x: x.shift(2))
#    ## After
#    data['word_type_1a'] = data.groupby(['doc_name'])['word_type'].apply(lambda x: x.shift(-1))
#    data['word_type_2a'] = data.groupby(['doc_name'])['word_type'].apply(lambda x: x.shift(-2))

    
    # Fillna Nan word shift
    data = data.fillna(-1)
    return data

def prediction(data, model):
    proba = model.predict(xgb.DMatrix(data[model.feature_names], missing=-1), ntree_limit=model.best_ntree_limit)
    prediction = [1. if y_cont > 0.5 else 0. for y_cont in proba] # binaryzing your output
    return prediction, proba
    
        

###


## Loading annexe files (Firstaname / Names etc;..)
# Reading French's firstnames file
firstname_df = pd.read_csv('data/prenom_clean.csv', encoding='utf-8',
                           header=None, names = ["firstname"])
# Use Majuscule in first carac
firstname_df['firstname'] = firstname_df['firstname'].apply(lambda x: x.title())
firstname_list = firstname_df.firstname.tolist()


# Reading foreign's firstnames file
foreign_firstname_df =  pd.read_csv('data/foreign_fistname_clean.csv', encoding='utf-8',
                             header=None, names = ["firstname"])
foreign_firstname_list = foreign_firstname_df.firstname.tolist()

# Name list
name_df = pd.read_csv('data/top_8k_name.csv', encoding='utf-8')
name_list = name_df.name.tolist()

# Stopword
stopword_fr = [word for word in stopwords.words('french')]

# stemmer
stemmer = SnowballStemmer("french")

mister_list = [u'M', u'M.', u'Madame', u'Mme', u'Monsieur', u'Dr', u'Monsieur', u'MM']


anymizer_list = [u"X",u"Y", u"Z", u"A", u"B", u"C", u"D", u"E", u"F",u"I",u"J",u"K", ]

#Bag of word
df_bag_of_word = pd.read_csv('data/encoding_label.csv', encoding='utf-8')
dico_word = dict(zip(df_bag_of_word.words, df_bag_of_word.encoded)) # Create a dico

#Type of word
word_type_info = pd.read_csv('data/word_type_info.csv', encoding='utf-8')
dico_word_type_info = dict(zip(word_type_info.word_type, word_type_info.encoded)) # Create a dico

# Loading model
f = open('model/stem_no_type_0.932578972183.model')
model = pickle.load(f)
f.close()


word_df = reading_docx('/Users/babou/github/paris_fellows/input/jf00151960.docx')
        
my_data = process_data(word_df)

prediction, proba = prediction(my_data, model)
my_data['prediction'] = prediction
my_data['proba'] = proba

# List of word positif
word_to_anonymse_list = my_data[my_data.prediction==1].word.unique().tolist()

to_anonymise_dict = {} 
for i in range(0, len(word_to_anonymse_list)):
    to_anonymise_dict[word_to_anonymse_list[i]] = anymizer_list[i]
                       
my_data['anonyme_word'] = my_data['word']
my_data.loc[my_data['prediction'] == 1, 'anonyme_word'] = my_data['word'].apply(lambda x: to_anonymise_dict.get(x, x))

# Si prediction ==1 et prediction n-1 == 1 alors on enlève la ligne sinon ==> M. X Y for Firstname / Name 
my_data['prediction_1b'] = my_data['prediction'].shift(1)


index_to_delete = my_data[(my_data['prediction'] == 1) & 
                            (my_data['prediction_1b'] == 1)].index.tolist()
                            
my_data = my_data.drop(index_to_delete)

print "Word to Anonymise :" 
for word in word_to_anonymse_list:
    print "--> " + str(word.encode('utf-8'))
    
# Write log file :
def get_proba_mean(x):
    """
    Return mean proba of word
    """
    return my_data[my_data.word == x].proba.mean()
    
def get_proba_min(x):
    """
    Return mean proba of word
    """
    return my_data[my_data.word == x].proba.min()
    
    

    
log_file = pd.DataFrame(word_to_anonymse_list, columns=['word'])
log_file['mean_proba_positif'] = log_file['word'].apply(lambda x: get_proba_mean(x))
log_file['min_proba'] = log_file['word'].apply(lambda x: get_proba_min(x))
log_file['warning'] = 0
log_file.loc[log_file.min_proba < 0.5, 'warning'] = 1


output = " ".join([word.encode('utf-8') for word in my_data.anonyme_word.tolist()])

name_ouput = my_data.doc_name.unique()[0]
text_file = open("output/"+name_ouput+".txt", "w")
text_file.write(output)
text_file.close()

log_file.to_csv("output/"+name_ouput+"_log.csv", index=False, encoding='utf-8')

def get_html_log(txt, log_file, to_anonymise_dict):
    """
    Return html page with color for warning name
    """
    
    html_code = '''<!DOCTYPE html><html><head>
        <meta charset="utf-8" />
        <title>Titre</title>
    </head><body>'''
    
    txt = unicode(txt.decode('utf-8'))
    #red color --> warning
    for hightlight_word in log_file[log_file.warning == 1].word:
        txt = txt.replace(hightlight_word, '''<font color="red"> '''+hightlight_word+'''</font>''')
        
    # Green color --> Ok
    for change_word in to_anonymise_dict.values():
        txt = txt.replace(" " + change_word + " ", '''<span style="background-color:#00ff00"> '''+" " +change_word+" "+'''</span>''')
        
    html_code = html_code + txt
    html_code = html_code + "</p></body></html>"
    return html_code.encode('utf-8')
    
output_html = get_html_log(output, log_file, to_anonymise_dict)

html_file = open("output/"+name_ouput+".html", "w")
html_file.write(output_html)
html_file.close()
        
        
    