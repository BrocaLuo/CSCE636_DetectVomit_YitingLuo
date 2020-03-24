import cv2
import os
import numpy as np

DIRECTORY= "C:\\Vomit\\Dataset"
f = open('C:\\Vomit\\Dataset\\Dataset.txt','r')
list=[]
for i in range(0,539):
    list.append(i)
num=[] 
imgs=[]
line=f.readline()
while line:
    a = line.split() 
    data = a[0]
    imgs.append(data)
    label = a[1]
    num.append(label)
    line = f.readline()
f.close()
 
batch=[]
labels=[]
for j in range(len(list)):
     num_1=list[j]
     file_path=DIRECTORY+"\\"+imgs[num_1]
     img=cv2.imread(file_path)
     img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
     batch.append(img)
     labels.append(num[num_1])
label1=[int(x)for x in labels]

def vectorize_sequences(sequences,dimension = 1024):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(batch)
y_train = np.asarray(label1).astype('float32')

DIRECTORY= "C:\\Vomit\\Test"
f = open('C:\\Vomit\\Test\\Test.txt','r')
list=[]
for i in range(0,142):
    list.append(i)
num=[] 
imgs=[]
line=f.readline()
while line:
    a = line.split() 
    data = a[0]
    imgs.append(data)
    label = a[1]
    num.append(label)
    line = f.readline()
f.close()
 
batch=[]
labels=[]
for j in range(len(list)):
     num_1=list[j]
     file_path=DIRECTORY+"\\"+imgs[num_1]
     img=cv2.imread(file_path)
     img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
     batch.append(img)
     labels.append(num[num_1])
label1=[int(x)for x in labels]

x_test = vectorize_sequences(batch)
y_test = np.asarray(label1).astype('float32')


