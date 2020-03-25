import cv2
import os
import numpy as np

DIRECTORY= "C:\\Vomit\\Dataset"
f = open('C:\\Vomit\\Dataset\\TrainResult.txt','r')
list=[]
for i in range(0,1004):
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
f = open('C:\\Vomit\\Test\\TestResult.txt','r')
list=[]
for i in range(0,203):
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



from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu',input_shape = (1024,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))



from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.0004),  loss='binary_crossentropy',   metrics=['accuracy'])

x_val = x_train[:600]
partial_x_train = x_train[600:]

y_val = y_train[:600]
partial_y_train = y_train[600:]


history = model.fit(partial_x_train,partial_y_train,epochs = 30,batch_size = 256,validation_data = (x_val,y_val))

import matplotlib.pyplot as plt
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values) + 1)

plt.plot(epochs,loss_values,'bo',label = 'Training loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf() 
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.plot(epochs,accuracy,'bo',label = 'Training accuracy')
plt.plot(epochs,val_accuracy,'b',label = 'Validation accuracy')

plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1024,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=256)
results = model.evaluate(x_test, y_test)

res=model.predict(x_test)
print(res)


from keras.models import load_model

model.save('my_model.h5')

model.save_weights('my_model_weights.h5')

