import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,BatchNormalization, MaxPooling2D
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback): 
	def on_train_begin(self, logs={}): 
		self.losses = {'batch':[], 'epoch':[]} 
		self.accuracy = {'batch':[], 'epoch':[]} 
		self.val_loss = {'batch':[], 'epoch':[]} 
		self.val_acc = {'batch':[], 'epoch':[]} 
	def on_batch_end(self, batch, logs={}): 
		self.losses['batch'].append(logs.get('loss')) 
		self.accuracy['batch'].append(logs.get('acc')) 
		self.val_loss['batch'].append(logs.get('val_loss')) 
		self.val_acc['batch'].append(logs.get('val_acc')) 
	def on_epoch_end(self, batch, logs={}): 
		self.losses['epoch'].append(logs.get('loss')) 
		self.accuracy['epoch'].append(logs.get('acc')) 
		self.val_loss['epoch'].append(logs.get('val_loss')) 
		self.val_acc['epoch'].append(logs.get('val_acc'))
	def loss_plot(self, loss_type): 
		iters = range(len(self.losses[loss_type]))
		plt.figure() 
		plt.plot(iters, self.losses[loss_type], 'g', label='train loss') 
		if loss_type == 'epoch': 
			plt.plot(iters, self.val_loss[loss_type], 'k', label='valid loss') 
		plt.grid(True) 
		plt.xlabel(loss_type) 
		plt.ylabel('loss') 
		plt.legend(loc="upper right") 
		plt.savefig('loss.jpg')
		plt.show()

	def acc_plot(self, loss_type): 
		iters = range(len(self.losses[loss_type]))
		plt.figure() 
		plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  
		if loss_type == 'epoch': 
			plt.plot(iters, self.val_acc[loss_type], 'b', label='valid acc') 
		plt.grid(True) 
		plt.xlabel(loss_type) 
		plt.ylabel('acc') 
		plt.legend(loc="upper left") 
		plt.savefig('accuracy.jpg')
		plt.show()

print(os.listdir("../input"))

import glob
train_imgs = []
train_label= []

train_dir = '../input/plant-seedlings-classification/train/*/*.png'

for img_dir in glob.glob(train_dir):
    img = Image.open(img_dir)
#     print("Label = " + img_dir.split('/')[-2] + " | for" + img_dir,img.format, img.size, img.mode)
#     print(img.resize((128, 128),Image.ANTIALIAS)) # ANTIALIAS to remove distortion, smoothening
    train_imgs.append(ImageOps.fit(img,(32, 32),Image.ANTIALIAS).convert('RGB'))
    train_label.append(img_dir.split('/')[-2])

images = np.array([np.array(im) for im in train_imgs])
images = images.reshape(images.shape[0], 32, 32, 3) / 255
lb = LabelBinarizer().fit(train_label)
label = lb.transform(train_label) 


trainX, validX, trainY, validY = train_test_split(images, label, test_size=0)

model = Sequential()
model.add(Conv2D(96, kernel_size=(3, 3),input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(96, kernel_size=(3, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(96, kernel_size=(3, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(12, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

history = LossHistory()

model.fit(trainX, trainY,
          batch_size=100,
          epochs=100,#run with 50 epochs first to get 95% accuracy
          validation_split = 0.05,
          callbacks=[history])

history.loss_plot('epoch')
history.acc_plot('epoch')

test_dir = '../input/plant-seedlings-classification/test/*.png'
test_imgs=[]
names = []
for timage in glob.glob(test_dir):
    img = Image.open(timage)
    names.append(timage.split('/')[-1])
    test_imgs.append(ImageOps.fit(img,(32, 32),Image.ANTIALIAS).convert('RGB'))

test_images = np.array([np.array(im) for im in test_imgs])
test_images_X = test_images.reshape(test_images.shape[0], 32, 32, 3) / 255

test_y = lb.inverse_transform(model.predict(test_images_X))

df = pd.DataFrame(data={'file': names, 'species': test_y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('final.csv', index=False)
