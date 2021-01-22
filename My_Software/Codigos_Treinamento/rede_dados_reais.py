import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import dan_lib

'''
Program writen by Daniel to control a simulated vehicle for lane keeping.

'''
plt.rcParams.update({'font.size':30})

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure
    plt.ylabel('Erro médio quadrático (RMS)')
    plt.xlabel('Epócas de treinamento')
    plt.plot(hist['epoch'],hist['loss'],label='Conjunto de Treinamento')
    plt.plot(hist['epoch'],hist['val_loss'], label = 'Conjunto de Validação')
    plt.legend()
    plt.grid()
    plt.show()


def sep_dir(nome):
    global left
    global right
    global straight
    #nome = nome[nome.find('_',48,len(nome)):len(nome)]
    #print("nome orig", nome)
    if nome.find('trecho', 1, len(nome)) == -1:
        #print("nomeori, ", nome)
        nome = nome[nome.find('r',25,len(nome)):len(nome)]
    else:
        nome = nome[nome.find('r', 25, len(nome)):len(nome)]
    #print("nome, ", nome)
    direita = nome[1:nome.find('l',1,len(nome))]
    #print("direita, ", direita)
    esquerda = nome[nome.find('l',1,len(nome))+1: nome.find('.', len(nome)-5, len(nome))]
    #print("esquerda, ", esquerda)
    direita = float(direita)
    if direita > 80:
        direita = 80
    if direita < -80:
        direita = -80
    esquerda = float(esquerda)
    if esquerda > 80:
        esquerda = 80
    if esquerda < -80:
        esquerda = -80

    if direita > esquerda:
        left = left + 1
        razao = -1+esquerda/direita
    elif direita <= esquerda:
        if direita != esquerda:
            right = right + 1
        if direita == esquerda:
            straight = straight + 1
        razao = 1 -direita/esquerda
    return razao

dir_list = ['new_data']
training_images = []
training_array = []
training_label = []

straight = 0
left = 0
right = 0

for dir in dir_list:
    path = '/home/daniel-lmi/'+dir+'/*.jpg'
    training_images.extend(glob.glob(path))

#print(len(training_images),len(training_images[0]))
#print(training_images)
time.sleep(2)

i = 0
display = False
t = 0
for images in training_images:
    img = cv2.imread(images)
    r = sep_dir(images)
    #print(float(r))
    img = cv2.resize(img,(192,112),cv2.INTER_AREA)
    #img = cv2.GaussianBlur(img,(5,5),0)
    #img_bin = dan_lib.hsv_select(img, (0,0,200), (179,17,255))
    #img[img_bin==0] = 50
    img = cv2.GaussianBlur(img,(5,5),0)
    blue,green,red = cv2.split(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if display:
        cv2.imshow('ibage', img)
        cv2.waitKey(10)
    training_array.append(green)
    training_label.append(r)
    if r > 0:
        right = right + 1
    elif r < 0:
        left = left + 1
    elif r == 0:
        straight = straight + 1
    i = i + 1

print(left, straight, right)

input("Deu ruim")

del training_images

training_array = np.array(training_array)
training_label = np.array(training_label)
training_array = np.reshape(training_array,[-1,112,192,1])
training_array = training_array.astype('float32')/255
print("shapes, ",training_array.shape,training_label.shape)
input("Deu ruim")

input_shape = (112,192,1)

batch_size = 128
kernel_size = 5
strides = 2
kernel_size2 = 3

model = Sequential()
model.add(Conv2D(filters = 24, kernel_size = kernel_size, strides = strides, activation = 'relu', input_shape = input_shape))
model.add(Conv2D(filters = 36, kernel_size = kernel_size, strides = strides, activation = 'relu'))
model.add(Conv2D(filters = 48, kernel_size = kernel_size, strides = strides, activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = kernel_size2, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters = 64, kernel_size = kernel_size2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('tanh'))
model.summary()
#time.sleep(50)

callback = tf.keras.callbacks.EarlyStopping(patience = 150, monitor = 'val_loss',restore_best_weights = True)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)

with tf.device('/gpu:0'):
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mean_squared_error'])

	history = model.fit(training_array, training_label, epochs = 5000, batch_size=batch_size, validation_split= 0.2, shuffle = False,callbacks=[callback])
	#model.fit(img_array, training_nlabel, epochs = 10000, batch_size=batch_size, validation_split=0.1, shuffle = True, callbacks = [callback])
loss= model.evaluate(training_array, training_label, batch_size=batch_size, verbose = 1)
plot_history(history)
print("terminei o treinamento")

os.chdir('/home/daniel-lmi')
model.save('my_model_realteste2.h5')
model.save('saved_model/my_model_real')
