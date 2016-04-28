# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:20:14 2016

@author: williamz
"""
import numpy as np
import pandas as pd
import string
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

lexicon=['side','effect','breast','studi','chemotherapi','hormon','lymph','love','famili','friend',
        'treatment','trial','clinic','surgeri', 'therapi', 'research', 'eat','meat','diet','food','drink',
        'oil','alcohol','veget','tumor','immun','system','mice','kill','stem','cell','hair','dye',
        'blond','brown','bald','wig','shave','color','ovarian','bladder','detect','oral','pink','young','survivor',
        'mammogram','skin','sun','melanoma','sunscreen','tan','mole','vitamin','exposur','dermatologist',
        'children','lung','smoke','prevent','vaccin','fruit','salad','pepper','fresh','green','bodi','consumpt',
        'fat','red','sugar','fish','target','cervic', 'hpv','sexual','women','viru','mom','dad','life','fear',
        'prostat','men','surviv','attitud', 'radiat','exercis','tamoxifen','aromatas','pill','treat','water','tea','green',
        'juic','tomato','tablespoon','oliv','blood','head','hat','curli','detect','symptom','earli','diseas','colorect','risk',
        'survivor','gardasil','infect','girl','pap','protect','human','mandatori','precancer','lump','ultrasound','calori',
        'cook','live','awar','urin','kid','psa','gay','protein','jude','st','hospit','donat','childhood','organ','lifestyl',
        'smoke','smoker','drug','menopaus','transplant','marrow','bone','race','leukemia','zometa','vegetarian','node','tan',
        'ray','sunburn','damag','shade','uvb','herceptin','gallbladd','brain','pain','spf','biopsi','testicular','heal','level',
        'mouth','gum','death','tobacco','tongu','tooth','laryng','dentist','vessel','angiogenesi','glioblastoma']
        
def text_process(s):
    s = s.translate(None, string.digits)
    s = s.lower()
    s = s.translate(None, string.punctuation)
    
    token_list = nltk.word_tokenize(s)
    STEMMER = PorterStemmer()
    stemming = [STEMMER.stem(tok.decode('utf-8',errors='ignore')) for tok in token_list]
    content = [w for w in stemming if w in lexicon]
    return ' '.join(content)
def randomforest_fit(x,y):
    tree = RandomForestClassifier(class_weight='auto')
    model = tree.fit(x,y)
    return model

print('Preparing data...')
train_all = pd.read_csv('frame_with_label.csv')

print('loading tags...')
tag_all = pd.read_csv('tag_all.csv',index_col='id')
tag_dic = [dict(zip(tag_all.index.tolist(),tag_all.iloc[:,i].tolist())) for i in range(6)]

print('training model..')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,1))
tfidf = tfidf_vectorizer.fit_transform(train_all.blog.tolist())

labels = [train_all.iloc[:,i].tolist() for i in range(1,7)]
models = [randomforest_fit(tfidf,i) for i in labels]

print('finished.\n')

file1 = raw_input('Enter file name to tag:')
text = open(file1,'r').read()
print('\nloading text..')
print len(text.split()),'words loaded'
text  = text_process(text)
test_data = tfidf_vectorizer.transform([text])

print ('fitting models...')
label_pre = [models[i].predict(test_data) for i in range(len(models))]

print('tagging the text file\n')
tags = [tag_dic[i][int(label_pre[i])] for i in range(6)]

print 'result:', list(set(tags)),'\n'

















