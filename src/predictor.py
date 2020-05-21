import numpy as np
import cv2
import glob
import os
import json
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf


face_cascade = cv2.CascadeClassifier('INPUT/haarcascade_frontalface_default.xml')

CLASSES = ['0-2','4-6','8-12','15-20', '21-35', '36-45', '46_59', '60_100']

def agePrediction(path, modelo_json, modelo_h5):
    with open (modelo_json,'r+') as f:
        model_json = json.load(f)
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(modelo_h5)
    im_bw = cv2.imread(path)
    im_bw = cv2.cvtColor(im_bw, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im_bw, 1.25, 6)
    x,y,w,h = faces[0]
    face = im_bw[y:y+h,x:x+w]
    face = cv2.resize(face,(48, 48))
    final_face = face.reshape(1,48, 48, 1)
    result = model.predict(final_face)
    itemindex = np.where(result == np.max(result))
    pred = (f"{str(round(np.max(result)*100))}% {CLASSES[itemindex[1][0]]}")
    return pred