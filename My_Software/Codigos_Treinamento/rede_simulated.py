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
import random
import dan_lib

'''
Program writen by Daniel based on https://github.com/priya-dwivedi/CarND/tree/master/CarND-Advanced%20Lane%20Finder-P4 to control a simulated vehicle for lane keeping recording its camera's image as a data-base for neural network training.

'''

plt.rcParams.update({'font.size':22})
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 22
#plt.rcParams['axes.labelweight'] = 'bold'

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure
    plt.ylabel('Erro médio quadrático')
    plt.xlabel('Epócas de treinamento')
    plt.plot(hist['epoch'],hist['loss'],label='train error')
    plt.plot(hist['epoch'],hist['val_loss'], label = 'val error')
    plt.title("Erro medio quadrático durante treinamento")
    plt.legend()
    plt.grid()
    plt.show()    

left = 0
right = 0
straight = 0
def sep_dir(nome):
    global left
    global right
    global straight
    #nome = nome[nome.find('_',48,len(nome)):len(nome)]
    print("nome orig", nome)
    if nome.find('trecho', 1, len(nome)) == -1:
        #print("nomeori, ", nome)
        nome = nome[nome.find('r',25,len(nome)):len(nome)]
    else:
        nome = nome[nome.find('r', 25, len(nome)):len(nome)]
    print("nome, ", nome)
    direita = nome[1:nome.find('l',1,len(nome))]
    print("direita, ", direita)
    esquerda = nome[nome.find('l',1,len(nome))+1: nome.find('.', len(nome)-5, len(nome))]
    print("esquerda, ", esquerda)
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
    return direita

#dir_list = ['curva1', 'curva1_inv','curva2','curva2_inv','curva3','curva3_inv','curva4','curva4_inv','curva5','curva5_inv','curva6','curva6_inv','curva7','curva7_inv','curva8','curva8_inv','curva9','curva9_inv','erro1','erro1_inv','erro2','erro2_inv','reta','reta_inv','volta','volta_inv']
#dir_list = ['training7', 'traning7/culva2', 'training7/culva12345', 'training7/fechado']
dir_list = ['propt']
training_images = []
#training_images = glob.glob('/home/daniel-lmi/training3/curva1/*.jpg')
training_array = []
training_label = []

for dir in dir_list:
    path = '/home/daniel-lmi/'+dir+'/*.jpg'
    training_images.extend(glob.glob(path))

#print(len(training_images),len(training_images[0]))
#print(left,straight,right)
#input("hehe")
time.sleep(2)

i = 0
display = False
for images in training_images:
    img = cv2.imread(images)
    r = sep_dir(images)
    print(float(r))
    if r > 0:
        right = right + 1
    elif r < 0:
        left = left + 1
    else:
        straight = straight + 1
    img = cv2.resize(img,(160,120),cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    '''
    if random.randint(1,101)<=70:
        img2 = img.copy()
        gauss = np.random.normal(0,1,(120,160,3))
        gauss = gauss.reshape(120,160,3)
        noisy = img2 + gauss
        training_array.append(noisy)
        training_label.append(r)
    else:
        img2 = img.copy()
        s_vs_p = 0.5
        amount = 0.004
        num_salt = np.ceil(amount*img2.size*s_vs_p)
        coords = [np.random.randint(0,i-1,int(num_salt))
                  for i in img2.shape]
        img2[coords] = 255
        num_pepper = np.ceil(amount*img2.size*(1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img2.shape]
        img2[coords] = 0
        #plt.imshow(img)
        #plt.imshow(img2)
        #plt.show()
    '''
    if display:
        cv2.imshow('ibage', img)
        cv2.waitKey(10)
    r = float(r)
    #if r < 0:
    #    r = -1 -r
    #if r>=0:
    #    r = 1 - r
    #print(r)
    training_array.append(img)
    training_label.append(r)
    i = i + 1

print(left,straight,right)
input("hehe")

del training_images

training_array = np.array(training_array)
training_label = np.array(training_label)
training_array = np.reshape(training_array,[-1,120,160,3])
training_array = training_array.astype('float32')/255
print("shapes, ",training_array.shape,training_label.shape)


input_shape = (120,160,3)

batch_size = 128
kernel_size = 5
strides = 2
kernel_size2 = 3

model = Sequential()
model.add(Conv2D(filters = 24, kernel_size = kernel_size, strides = strides, activation = 'relu', input_shape = input_shape))
model.add(Conv2D(filters = 36, kernel_size = kernel_size, strides = strides, activation = 'relu'))
model.add(Conv2D(filters = 48, kernel_size = kernel_size, strides = strides, activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = kernel_size2, activation = 'relu'))
model.add(Dropout(0.2))
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
model.summary()
#time.sleep(50)

callback = tf.keras.callbacks.EarlyStopping(patience = 100, monitor = 'val_loss',restore_best_weights = True)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)

with tf.device('/gpu:0'):
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mean_squared_error'])

	history = model.fit(training_array, training_label, epochs = 450, batch_size=batch_size, validation_split= 0.2, shuffle = True,callbacks=[callback])
	#model.fit(img_array, training_nlabel, epochs = 10000, batch_size=batch_size, validation_split=0.1, shuffle = True, callbacks = [callback])
loss= model.evaluate(training_array, training_label, batch_size=batch_size, verbose = 1)
plot_history(history)
print("terminei o treinamento")

os.chdir('/home/daniel-lmi')
#model.save('my_model_prop.h5')
#model.save('saved_model/my_model_prop')
'''
for i in range(1,500):

	img = np.reshape(training_array[i],[-1,100,320,3])
	#img = training_array[i]
	#img = img.astype('float32')/255
	print("Prediction: ", model.predict(img), " Label: ", training_label[i])
	#print(model.predict(img))
	#print("Label")
	#print(training_label[i])



#model.save('rede_pcdan.h5')
'''
