from joblib import load
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from functools import cached_property
from joblib import load

def predict_image(img_path):
    image = imread(img_path)
    image = resize(image, (128, 128))
    image = image.flatten().reshape(1, -1)

    classifier = load('classifier.joblib')

    return classifier.predict(image)