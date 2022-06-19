# Basic Flask Server
from flask import Flask, render_template
#from flask_sqlalchemy import SQLAlchemy -> will use this for our db

app = Flask(__name__)

from aiart import routes