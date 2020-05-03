import cv2
from keras.models import load_model
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

#load model
model = load_model('./cnn.model')
lb = pickle.loads(open('./cnn_lb.pickle', "rb").read())
fileJson = {}
#change the name of the video if needed
fileJson["2.mp4"] = []
model.summary()
import cv2
#change the name of the video if needed
vc=cv2.VideoCapture("2.mp4")
c=1
X1 = []
Y1 = []

#Prediction
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
#change the name of the video if needed
      fileJson["2.mp4"].append([str(c),str([preds[0][0],preds[0][1]])])
      X1.append([c])
      Y1.append([preds[0][1]])
      c=c+1
      cv2.waitKey(1)
vc.release()

import json

with open("File.json", "w") as outfile: 
    json.dump(fileJson, outfile)
#Plot the time label image
fileJson
X1 = np.array(X1)
Y1 = np.array(Y1)
for i in range(len(X1)):
  plt.plot(X1[i], Y1[i], color='r')
  plt.scatter(X1[i], Y1[i], color='b')
#fig = plt.step(c, preds[0][1])
plt.xlabel('Video length (in terms of frame number)')
plt.ylabel('Prediction')
#plt.xticks(np.arange(0, video_length,round(frame_step_size*time_per_frame,2) ))

plt.show()




