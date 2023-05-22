from flask import Flask, request, render_template
#from classifier_training import train_classifier
from predict_image import predict_image
import os
import numpy as np
import torch as tc
import torchvision as tv
import SimpleITK as sitk

app = Flask(__name__)
MODEL_PATH = 'classifier.joblib'
'''
def load_classifier():
    if os.path.isfile(MODEL_PATH):
        return load(MODEL_PATH)
    else:
        return train_classifier()
    
def remove_file(file_path):
    os.remove(file_path)

# Uczymy klasyfikatora po pierwszym uruchomieniu aplikacji
@app.before_first_request
def train_on_startup():
    load_classifier()
'''
# Obsługa formularza
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/images/' + file.filename
        file.save(file_path)
        prediction = predict_image(file_path)

        if prediction == 0:
            prediction_msg = "Krwotok wykryty"
        else:
            prediction_msg = "Nie wykryto krwotoku"

        return render_template('result.html', prediction=prediction_msg, image_file = file.filename)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)