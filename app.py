from flask import Flask, request, render_template, url_for, flash, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, IntegerField, ValidationError
from wtforms.validators import Required

import requests
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardtoguessstring'

class WeatherEntryForm(FlaskForm):
    zip= StringField("Enter Zip code:", validators=[Required()])
    submit = SubmitField('Submit')

    def validate_zip(self, field):
        if len(field.data) != 5:  # REPLACE THIS LINE WITH SOMETHING ELSE
            raise ValidationError("Your zipcode did not contain 5 integers")


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/zipcode')
def weather_form():
    weather = WeatherEntryForm()
    return render_template("weather_form.html", form=weather)

@app.route('/weather_result', methods=['GET', 'POST'])
def show_results():
    form = WeatherEntryForm()
    if request.method == "POST" and form.validate_on_submit():
        zip = form.zip.data
        r = requests.get('http://api.openweathermap.org/data/2.5/weather?zip=' + zip + ',us&appid=acb8954b723c21c5d82de826a487e67a')
        r = r.json()
        city = r["name"]
        description = r['weather'][0]['description']

        temp = 9/5 * (r['main']['temp'] - 273) + 32

        return render_template('weather_results.html', city = city, description= description, temp = temp)
    flash(form.errors)
    return redirect(url_for('weather_form'))

if __name__ == "__main__":
    app.run()