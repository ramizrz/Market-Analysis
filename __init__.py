from flask import Flask
from flask.ext.images import resized_img_src


app = Flask(__name__)
app.config['SECRET_KEY'] = 'Msg4AwyDdMEcG7PHcrqlcA'

from flaskblog import app
