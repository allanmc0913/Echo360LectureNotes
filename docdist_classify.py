import csv
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


#########################
##### Doc/Dist Code #####
#########################
def doc_dist(question):
    reader = csv.DictReader(open("qa.csv", 'r', encoding="latin-1"),
                            fieldnames=["questionBody", "questionResponse"])

    questions = []
    answers = []

    for qa in reader:
        questions.append(qa['questionBody'])
        answers.append(qa['questionResponse'])

    # breaks up all the words/punctuation in each question into their own list (aka tokenizes)
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in questions]

    # for each entry, it maps each word to a number
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

    # do all of{ c2_tag }} the same above for the query you want to make
    query_doc = [w.lower() for w in word_tokenize(question)]

    query_doc_bow = dictionary.doc2bow(query_doc)

    query_doc_tf_idf = tf_idf[query_doc_bow]

    s = sims[query_doc_tf_idf]

    maxnum = s.max()

    answer = answers[int(np.argmax(s))]

    closest_question = questions[int(np.argmax(s))]

    return closest_question, answer, maxnum


if __name__ == "__main__":

    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    with open("naivebayes.pickle", 'rb') as f:
        classify = pickle.load(f)

    while True:
        q = input("\n\nEnter a question: ")
        closest, answer, percent = doc_dist(q)

        qtype = classify.predict([q])

        print("\nThis is classified as a course {} question.".format(qtype[0]))
        print("The closest answer we have for your question is:")
        print("\n\tQ: {}\n\tA: {}".format(closest, answer))
        print("With a {}% match rate".format(int(percent*100)))
        print("---------------------")
