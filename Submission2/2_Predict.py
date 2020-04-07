import cv2
from keras.models import load_model
import argparse
import pickle
import cv2

model = load_model('./cnn.model')
lb = pickle.loads(open('./cnn_lb.pickle', "rb").read())
fileJson = {}
fileJson["Vomit10.mp4"] = []

import cv2
vc=cv2.VideoCapture("Vomit10.mp4")
c=1
if vc.isOpened():
      rval,frame=vc.read()
else:
      rval=False
while rval:
      cv2.imwrite('C:\\Vomit\\PredictSet\\Predict'+str(c)+'.jpg',frame)
      rval,frame=vc.read()
      image = cv2.imread('C:\\Vomit\\PredictSet\\Predict'+str(c)+'.jpg')
      output = image.copy()
      image = cv2.resize(image, (64, 64))
      image = image.astype("float") / 255.0
      image = image.reshape((1, image.shape[0],
      image.shape[1],image.shape[2]))
      preds = model.predict(image)
      i = preds.argmax(axis=1)[0]
      label = lb.classes_[i]
      text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
      fileJson["Vomit10.mp4"].append([str(c),str([preds[0][0],preds[0][1]])])

      c=c+1
      cv2.waitKey(1)
vc.release()

import json

with open("File.json", "w") as outfile: 
    json.dump(fileJson, outfile)

fileJson






