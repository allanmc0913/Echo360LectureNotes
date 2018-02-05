from flask import Flask, request, render_template, url_for, flash, redirect, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, IntegerField, ValidationError
from wtforms.validators import Required



import requests
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardtoguessstring'



@app.route('/')
def hello_world():
    return 'Hello World!'



if __name__ == '__main__':
    app.run()



