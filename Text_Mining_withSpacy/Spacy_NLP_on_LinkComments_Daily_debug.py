#!/usr/bin/env python
# coding: utf-8
# %%


import spacy
#import el_core_news_sm
import string
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from azureml.core import Experiment
from azureml.core import Workspace, Dataset
from azureml.data import DataType
from spacy.cli.download import download as spacy_download



# %%
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '6ed9d167-b2e6-41b8-9500-35e6df64d9dc'
resource_group = 'MLRG'
workspace_name = 'erbbimlws'

workspace = Workspace(subscription_id, resource_group, workspace_name)


datastore = workspace.get_default_datastore()



# %%
spacy_download('el_core_news_sm')
nlp =spacy.load('el_core_news_sm', disable=['tagger', 'parser', 'ner'])


# %%


p1 = re.compile('δεν απαντ.{1,3}\s{0,1}',re.IGNORECASE)
p2 = re.compile('\sδα\s',re.IGNORECASE)
p3 = re.compile('δε.{0,1}\s.{0,3}\s{0,1}βρ.{1,2}κ.\s{0,1}',re.IGNORECASE)
p4 = re.compile('[^\d]?\d{10}')
p5 = re.compile('[^\d]?\d{18}|[^\d]\d{20}')
p6 = re.compile('δε[ ν]{0,1} (επιθυμ[α-ω]{2,4}?|[ήη]θ[εέ]λ[α-ω]{1,3}?|θελ[α-ω]{1,4}|.{0,20}ενδιαφ[εέ]ρ[α-ω]{2,4})',re.IGNORECASE)
p7 = re.compile('δε[ ν]{0,1} (μπορ[α-ω]{2,5}|.εχει)',re.IGNORECASE)
p8 = re.compile('(δεν|μη).*διαθεσιμ[οη]ς{0,1}?',re.IGNORECASE)
p9 = re.compile('(δεν|μη)+.*εφικτη?',re.IGNORECASE)
p10 = re.compile('δε[ ν]{0,1}.{0,20}θετικ[οόήη]ς{0,1}',re.IGNORECASE)


# %%


def loadStopWords():
    dataset = Dataset.get_by_name(workspace, name='stopWords_gr')
    sw = set(dataset.to_pandas_dataframe())
    return sw


# %%


def replaceTerm(text):
    text = p5.sub(' λογαριασμός ',text)
    text = p4.sub(' τηλεφωνο ',text)
    text = p6.sub(' δενθελειδενενδιαφερεται ',text)
    text = p10.sub(' δενθελειδενενδιαφερεται ',text)
    text = p7.sub(' δενεχειδενμπορει ',text)
    text = p8.sub(' δενειναιδιαθεσιμος ',text)
    text = p9.sub(' ανεφικτη ',text)
    text = text.replace('-banking','banking')
    text = text.replace('v banking','vbanking')
    text = text.replace('e banking','ebanking')
    text = text.replace('follow up','followup')
    text = text.replace('fup','followup')
    text = text.replace('f/up','followup')
    text = text.replace('πυρ/ριο','πυρασφαλιστηριο')
    text = text.replace('safe drive','safedrive')
    text = text.replace('safe pocket','safepocket')
    text = text.replace('alphabank','alpha')
    text = text.replace('sweet home smart','sweethomesmart')
    text = text.replace('sweet home','sweethome')
    text = text.replace('eξασφαλιζω','εξασφαλιζω')
    text = text.replace('credit card','creditcard')
    text = text.replace('debit card','debitcard')
    text = text.replace('life cycle','lifecycle')
    text = text.replace('π/κ','πκ')
    text = text.replace('td','πκ')
    text = text.replace('α/κ','ακ')
    text = text.replace('δ/α','δεναπαντα ')
    text = text.replace('εκτος αττικης','εκτοςαττικης ')
    #τδ
    text = p1.sub(' δεναπαντα ',text)
    text = p2.sub(' δεναπαντα ',text)
    text = p3.sub(' δεντονβρηκα ',text)
    
    return text


# %%


sw = loadStopWords()
def remove_ton(text):
    diction = {'ά':'α','έ':'ε','ί':'ι','ό':'ο','ώ':'ω','ύ':'υ'}
    for key in diction.keys():
        text = text.replace(key, diction[key])
    return text   
def clean_text(text):
     #text to string
    text = str(text).lower()
    text = replaceTerm(text)
    
   # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # lower text
    text = [remove_ton(x) for x in text]
    # remove stop words
    text = [x for x in text if x not in sw]
 
    #remove quotes
    text = [x.replace('quot;','').replace('&quot','') for x in text if x not in {'quot','amp'}]
    # remove words that contain numbers
    #text = [word for word in text if not any(c.isdigit() for c in word)] #addition to return even empty tokens
    # remove empty tokens
    #text = [t for t in text if len(t) > 0] #addition to return even empty tokens
    # remove amp & quot
    text = [x for x in text if x not in ['quot','amp']]
    # remove words with only one letter
    text = " ".join([t for t in text if len(t) > -1]) #addition to return even empty tokens
     # lemmatize text
    text = " ".join([t.lemma_ for t in nlp(text, disable=['tagger', 'parser', 'ner','tok2vec', 'morphologizer', 'parser', 'senter', 'attribute_ruler',  'ner'])])
   
    return(text)


# %%


def load_correctDict():
        
    dataset = Dataset.get_by_name(workspace, name='correct_Tokens')    
    corDict = dict(dataset.to_pandas_dataframe().to_dict("split")['data'])
    return corDict


# %%


def correct(x,corDict):

    if x in corDict.keys():
        y = corDict[x]
    else:
        y = x
    return y    


# %%


def get_ngrams(idf,mindf,minngram,maxngram):
    tfidf = TfidfVectorizer(min_df = mindf,ngram_range = (minngram,maxngram))
    tfidf_result = tfidf.fit_transform(idf['tokenized']).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = [str(x) for x in tfidf_df.columns]
    df_i = pd.concat([df[['CON_ROW_ID']],tfidf_df],axis=1).melt(id_vars=['CON_ROW_ID'],value_vars = tfidf_df.columns).dropna()
    df_i = df_i[df_i['value']>0]
    return df_i


# %%


def cleanComments(df):
    df = df[['CON_ROW_ID','CON_COMMENTS']]
    df['tokenized'] = df['CON_COMMENTS'].apply(clean_text)
    df = df.fillna('N/A')
    df['variable'] = df['tokenized'].str.split()
    return df


# %%


def getTokens(df):
    df = cleanComments(df)
    df_f = df.explode('variable')[['CON_ROW_ID','variable']]
    return df_f


# %%


def getTokencount(df_f,minCount):
    tokenCount = df_f['variable'].value_counts().to_dict()
    df_f['value'] = df_f['variable'].map(tokenCount)
    
    df_f.loc[(df_f['value']<minCount), 'variable'] = ' ' #addition to return even empty tokens
    return df_f


# %%
txt = 'AYJHSE SXESH? POYLHSE AKINHTO?'
com = {'CON_ROW_ID':[1],'CON_COMMENTS':[txt]}
df = pd.DataFrame(com)


# %%


df = cleanComments(df)


# %%


df_f = getTokens(df)


# %%
df_f = df_f.fillna(' ')

# %%
df_f.head()

# %%
minCount = 30


# %%



df_f = getTokencount(df_f,minCount)


# %%
df_f.head()

# %%


#ngrams parameters
mindf,minngram,maxngram = 1000,2,3


# %%
try:
    df_f = df_f.append(get_ngrams(df,mindf,minngram,maxngram ))
except:
    print('no bigramms or trigramms were added')


# %%


corDict = load_correctDict()


# %%


df_f['token'] = df_f['variable'].apply(lambda x : correct(x,corDict))


# %%
df_f.head()




# %%

df_f.loc[(df_f['token'].str.len() <2), 'token'] = ' ' #addition to return even empty tokens
#df_f = df_f[((df_f['token'].str.len() >1) | (df_f['token']==' ' ))]



# %%


df_f = df_f.fillna('N/A')


# %%

# %%


df_f = df_f.sort_values(['CON_ROW_ID','token'])


# %%


df_f = df_f[['CON_ROW_ID','token']].drop_duplicates()


# %%
df_f

# %%


#df_f['token'].value_counts().to_excel('./xlsx/tokenlist_new.xlsx')


# %%


#run.complete()

