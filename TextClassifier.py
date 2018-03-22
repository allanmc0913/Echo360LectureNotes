
# coding: utf-8

# In[10]:


import pandas as pd


full_data = pd.read_csv("qadata.csv", encoding="latin-1")
full_data.head()


# In[68]:


full_data = full_data[["questionBody", "questionType"]]
full_training = full_data[(full_data.questionType == "logistic") | (full_data.questionType == "content")]
training = full_training.sample(frac=0.75)
training.head()


# In[55]:


categories = ['content', 'logistic']


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training.questionBody)
X_train_counts.shape


# In[57]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[58]:


# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, training.questionType)


# In[59]:


# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(training.questionBody, training.questionType)


# In[60]:


# Performance of NB Classifier
import numpy as np
predicted = text_clf.predict(full_training.questionBody)
np.mean(predicted == full_training.questionType)


# In[61]:


# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(training.questionBody, training.questionType)
predicted_svm = text_clf_svm.predict(full_training.questionBody)
np.mean(predicted_svm == full_training.questionType)


# In[62]:


# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}


# In[63]:


# Next, we create an instance of the grid search by passing the classifier, parameters 
# and n_jobs=-1 which tells to use multiple cores from user machine.

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(training.questionBody, training.questionType)


# In[64]:


# To see the best mean score and the params, run the following code

gs_clf.best_score_
gs_clf.best_params_


# In[65]:


# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(training.questionBody, training.questionType)


gs_clf_svm.best_score_
gs_clf_svm.best_params_


# In[66]:


# NLTK
# Removing stop words
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])


# In[67]:


# Stemming Code
import nltk

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(training.questionBody, training.questionType)

predicted_mnb_stemmed = text_mnb_stemmed.predict(full_training.questionBody)

np.mean(predicted_mnb_stemmed == full_training.questionType)


# In[69]:


to_predict = full_data[(full_data.questionType != "logistic") & (full_data.questionType != "content")]


# In[70]:


text_to_predict = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=False))])

text_to_predict = text_to_predict.fit(full_training.questionBody, full_training.questionType)

predicted_to_predict = text_to_predict.predict(to_predict.questionBody)


# In[73]:


to_predict['questionType'] = predicted_to_predict


# In[75]:


to_predict

