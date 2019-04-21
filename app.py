from flask import Flask, render_template
#from flask.ext.images import resized_img_src
#from PIL import images
import kmeans
import hc
app= Flask(__name__)
app.config['SECRET_KEY'] = 'Msg4AwyDdMEcG7PHcrqlcA'

#images=Images(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/contact')
def contact():
	return render_template('contact.html')


@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/visual')
def visual():
	return render_template('visual.html',hc.target())

@app.route('/analyse')
def analyse():
	 return render_template('analyse.html',kmeans.target())

if __name__=='__main__':
 app.run(debug=True)
