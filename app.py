from flask import Flask, request, render_template, url_for, flash, redirect, session, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, IntegerField, ValidationError, TextAreaField, PasswordField
from wtforms.validators import Required
from flask_sqlalchemy import SQLAlchemy
import sys
import logging

from flask_script import Manager, Shell
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

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


app.config["SQLALCHEMY_DATABASE_URI"] = "postgres://ctextsoopbzvfh:f0569e5505fe6e3de7c1fcc08485a505865315a2593eadb3020c54c7a7e3f136@ec2-50-17-206-214.compute-1.amazonaws.com:5432/df76ak8ajgqgrr"
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
db.Model.metadata.reflect(db.engine)
manager = Manager(app)








##ORM SQLAlchemy DB Model
class question_answer(db.Model):
    __table__ = db.Model.metadata.tables["question_answer"]

    #__table_args__ = {'autoload': True, 'autoload_with': engine}

    def __repr__(self):
        return "Question: {}, Answer: {})".format(self.question, self.answer)

class key_def(db.Model):
    __table__ = db.Model.metadata.tables["keyterm_definitions"]

    def __repr__(self):
        return "Keyterm: {}, Definition: {})".format(self.keyword, self.definition)

class unanswered_question(db.Model):
    __tablename__ = "unanswered_questions"
    __table_args__ = {'extend_existing':True}
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, unique=True)
    answer = db.Column(db.Text)


##Flask-WTForm asking for user input
class Question(FlaskForm):
    question = StringField("")
    submit = SubmitField('Submit')


class Unanswered(FlaskForm):
    a1 = TextAreaField()
    a2 = TextAreaField()
    a3 = TextAreaField()
    a4 = TextAreaField()
    a5 = TextAreaField()
    submit = SubmitField('Submit')

class Authentication(FlaskForm):
    pw = PasswordField("GSI/Professor, Please enter password: ", validators=[Required()])
    submit = SubmitField()

    def validate_pw(self, field):
        if field.data != "echo360":
            raise ValidationError("You have entered in the wrong password")


class AddNew(FlaskForm):
    q1 = StringField("Add question:", validators=[Required()])
    a1 = StringField("Add answer:", validators=[Required()])
    submit = SubmitField()



#########################
##### Doc/Dist Code #####
#########################

######### Doc Dist for Question/Answers###############
def doc_dist(question):

    questions = []
    answers = []

    for q in question_answer.query.all():
        questions.append(q.question)
    questions.pop(0)

    for a in question_answer.query.all():
        answers.append(a.answer)
    answers.pop(0)


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

    max = s.max()

    answer = answers[int(np.argmax(s))]

    closest_question = questions[int(np.argmax(s))]

    return closest_question, answer, max


########## Doc Dist for Keywords_Definitons##################
def k_doc_dist(keyword):

    keyterms = []
    definitions = []

    for k in key_def.query.all():
        keyterms.append(k.keyword)
    keyterms.pop(0)

    for d in key_def.query.all():
        definitions.append(d.definition)
    definitions.pop(0)

    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in keyterms]

    dictionary = gensim.corpora.Dictionary(gen_docs)

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

    tf_idf = gensim.models.TfidfModel(corpus)
    s = 0
    for i in corpus:
        s += len(i)

    sims = gensim.similarities.MatrixSimilarity(tf_idf[corpus],
                                                num_features=len(dictionary))

    query_doc = [w.lower() for w in word_tokenize(keyword)]

    query_doc_bow = dictionary.doc2bow(query_doc)

    query_doc_tf_idf = tf_idf[query_doc_bow]

    s = sims[query_doc_tf_idf]

    max_score_k = s.max()

    definition = definitions[int(np.argmax(s))]

    closest_keyterm = keyterms[int(np.argmax(s))]

    return closest_keyterm, definition, max_score_k




##Display form
@app.route('/')
def question():
    form = Question()
    return render_template("question_form.html", form=form)


##Show inputted form question
@app.route('/question_result', methods=['GET', 'POST'])
def show_results():
    form = Question()
    unanswered_lst = []

    if request.method == "POST":

        ## create a new list to take in questions that dont have answer in db, for every one question
        ## create new view function to take in list
        ## create new template HTML
        ## update db

        question = form.question.data
        closet_question, answer, max_score = doc_dist(question)  #qa
        closest_keyterm, definition, max_score_k = k_doc_dist(question)  #kd


        if max_score < max_score_k: #if qa < kd
            closet_question = closest_keyterm
            question = question
            answer = definition

        if max_score_k < max_score:  #if kd < qa
            closet_question = closet_question
            question = question
            answer = answer

        if (max_score < 0.5) and (max_score_k < 0.5):
            search_term = question.replace(" ", "+")
            search_url = "https://www.google.com/search?q=" + str(search_term)
            new = unanswered_question(question=question)
            db.session.add(new)
            db.session.commit()
            # unanswered_lst.append(question)
            # session['unanswered_questions'] = unanswered_lst

            return render_template('return_question.html', question=question, search=search_url, form=form)

        return render_template('return_question.html', closest_question=closet_question, question=question, answer=answer, form=form)
    flash(form.errors)
    return redirect(url_for('question'))


@app.route('/staff_login', methods=['GET', 'POST'])
def staff_login():
    form = Authentication()
    session['is_staff'] = False
    if request.method == "POST" and form.validate_on_submit():
        session['is_staff'] = True
        return redirect(url_for("add_question_answer"))

    errors = [v for v in form.errors.values()]
    if len(errors) > 0:
        flash("!!!! ERRORS IN FORM SUBMISSION - " + str(errors))
    print (form.errors)
    return render_template('staff_login.html', form=form)


@app.route('/add_question_answer', methods=['GET', 'POST'])
def add_question_answer():
    form = AddNew()

    is_staff = session.get('is_staff', None)

    if is_staff is True and request.method == "POST" and form.validate_on_submit():

        q1 = form.q1.data
        a1 = form.a1.data
        new = question_answer(question=q1, answer=a1)
        db.session.add(new)
        db.session.commit()
        return redirect(url_for('add_question_answer'))

    return render_template("add_question_answer.html", form=form, is_staff=is_staff)

@app.route('/unanswered_questions', methods=['GET', 'POST'])
def unanswered_questions():
    unanswered_lst = unanswered_question.query.filter_by(answer=None).all()
    question_lst = []
    is_staff = session.get('is_staff', None)

    for item in unanswered_lst:
        question = item.question
        question_lst.append(question)

    if request.method == "GET" and is_staff is True:
        d = dict(request.args)
        e = list(d.values())
        for answer in e:
            q_a = list(zip(question_lst, answer))

            for pair in q_a:
                result = unanswered_question.query.filter_by(question=pair[0]).first()
                if result is not None:
                    result.answer = pair[1]
                    db.session.commit()
                    return redirect(url_for("unanswered_questions"))

    return render_template("unanswered_questions.html", question_lst=question_lst, is_staff=is_staff)


if __name__ == '__main__':
    db.create_all()
    app.run(use_reloader=True, debug=True)
    #manager.run()
