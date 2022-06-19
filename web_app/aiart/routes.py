from aiart import app
from flask import render_template

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html', item_name = 'test to see how data can be rendering from here to about.html')