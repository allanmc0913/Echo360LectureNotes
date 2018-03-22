from flask import Flask, request, render_template, url_for, flash, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, IntegerField, ValidationError
from wtforms.validators import Required
from flask_sqlalchemy import SQLAlchemy


import requests
import json
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
import numpy as np


#########################
##### App/DB Setup #####
#########################

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardtoguessstring'
app.debug=True
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://localhost/echo360"
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)



def doc_dist(question):
    df = pd.read_csv('/Users/AllanChen/Desktop/MVP/ALPquestionsTXT_20180201.csv', encoding='latin-1')

    qa = df[['questionBody', 'questionResponse']]

    temp = qa['questionBody']
    questions = []
    for r in temp:
        questions.append(r)





    #breaks up all the words/punctuation in each question into their own list (aka tokenizes)
    gen_docs = [[w.lower() for w in word_tokenize(text)]
                for text in questions]



    #for each entry, it maps each word to a number
    dictionary = gensim.corpora.Dictionary(gen_docs)

    # creates tuple pairs of the word(their mapped number) and how many times they appear in the document
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]


    # tf-idf model of (num of entries, num of tokens/words/punc)
    tf_idf = gensim.models.TfidfModel(corpus)
    s = 0
    for i in corpus:
        s += len(i)

    # matrix similarity: https://stackoverflow.com/questions/36578341/how-to-use-similarities-similarity-in-gensim
    sims = gensim.similarities.MatrixSimilarity(tf_idf[corpus],
                                          num_features=len(dictionary))


    # do all of the same above for the query you want to make
    query_doc = [w.lower() for w in word_tokenize(question)]

    query_doc_bow = dictionary.doc2bow(query_doc)

    query_doc_tf_idf = tf_idf[query_doc_bow]

    s = sims[query_doc_tf_idf]

    answer = qa['questionResponse'][np.argmax(s)]
    closest_question = qa['questionBody'][np.argmax(s)]

    return closest_question, answer
