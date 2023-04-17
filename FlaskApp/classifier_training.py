import joblib
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

def train_classifier():
    input_dir = "C:\\Users\\Lenovo\\Desktop\\TOM\\CT_manual\\test"
    categories = ['Homm', 'No_homm']

    data = []
    labels = []

    for category_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join(input_dir, category)):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            img = resize(img, (128, 128))
            data.append(img.flatten())
            labels.append(category_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, shuffle = True, stratify = labels)

    classifier = SVC()

    parameters = [{'gamma': [0.0001, 0.001, 0.01], 'C': [6, 8, 10, 20]}]

    grid_search = GridSearchCV(classifier, parameters)

    grid_search.fit(x_train, y_train)

    best_estimator = grid_search.best_estimator_

    joblib.dump(best_estimator, 'classifier.joblib')