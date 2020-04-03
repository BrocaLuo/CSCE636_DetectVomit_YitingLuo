import cv2
import os
import numpy as np
import skvideo
import keras
from glob import glob
import os
from keras.layers import Lambda, concatenate
from keras import Model
import tensorflow as tf

videoPath= "C:\\Vomit\\Vomit7.mp4"

from keras.models import load_model

model = load_model('C:\\Vomit\\my_model.h5')
model.summary()
#test = np.zeros(1024,)
test = []
#test = np.expand_dims(test,axis = 0)
#predict = model.predict(test)
#predict[0]
testFrames = []
#testFrames.shape
skvideo.setFFmpegPath(r'C:\\Vomit\\ffmpeg-4.2.2-win64-static\\bin')
import skvideo.io

#testFrames = skvideo.io.vread(videoPath, height = 224, width = 224)
testFrames = skvideo.io.vread(videoPath, outputdict={"-pix_fmt": "gray"})[:, 224, 224, 0]
i = 0
fileJson = {}
fileJson["Vomit7.mp4"] = []
print (testFrames.shape)

while(i<testFrames.shape[0]):
  #test = testFrames[i:i+5,:,:,:]
  
  test = testFrames[i,:,:]
  print (test.shape)
  testdata = np.expand_dims(test, axis = 0)
  predict = model.predict(test)
  predict = model.predict(np.array(testdata))
  fileJson["Vomit7.mp4"].append([str(i),str([predict[0][0],predict[0][1]]) ] )
  i += 5
import json

with open("File.json", "w") as outfile: 
    json.dump(fileJson, outfile)

fileJson