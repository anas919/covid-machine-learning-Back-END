import os
from flask import Flask, render_template, request, redirect, url_for,jsonify,flash
import base64
from flask_cors import CORS
#New Libraries
from werkzeug.utils import secure_filename
import pickle
import cv2
import keras
from tensorflow.keras.models import load_model
from keras import backend as K

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
cors = CORS(app)


@app.route('/')
def index():
	return render_template('index.html')

def predict_img(url):
	arr = cv2.imread(url)
	arr = cv2.resize(arr, (100, 100))
	# print(arr)
	# print('==========================================')
	arr = arr.reshape(-1, 100, 100, 3)
	# print(arr)
	arr = arr/255
	# print('==============')
	# print(arr)
	K.clear_session()
	model_dir = os.path.realpath('./models')
	model_path = model_dir + '/pnemonia_prediction.model'
	model = load_model(model_path)
	CLASS = ['NORMAL','PNEUMONIA']
	prediction = model.predict(arr)
	# print(prediction)
	# print(prediction.argmax())
	return CLASS[prediction.argmax()]

@app.route('/show', methods=['POST', 'GET'])
def trait_image():
	K.clear_session()
	model_dir = os.path.realpath('./models')
	model_path = model_dir + '/pnemonia_prediction.model'
	model = load_model(model_path)
	#File Code
	if request.method == 'POST':
		uploaded_file = request.files['image_to_trait']
		if uploaded_file.filename != '':
			uploaded_file.save(uploaded_file.filename)
			result = predict_img(uploaded_file.filename)
			# print(result)

	# return index(result)
	return jsonify(
        result=result,
    )
    #"result":"NORMAL"

if __name__ == '__main__':
	app.run()