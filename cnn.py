import numpy as np 
import pandas as pd 
#import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings #on tanimli olarak yuklu gelmesi lazım
# filter warnings
warnings.filterwarnings('ignore')

train= pd.read_csv(r'fashion-mnist_train.csv') #dosyalari okumak
test = pd.read_csv(r'fashion-mnist_test.csv')

# etiketleri y degiskenine koyuyoruz
Y_train = train["label"]
# etiket kolonu
X_train = train.drop(labels = ["label"],axis = 1)

# etiketleri y degiskenine koyuyoruz
Y_test = test["label"]
# etiket kolonu
X_test = test.drop(labels = ["label"],axis = 1) 

# dataları normalize ediyoruz
X_train = X_train / 255.0
X_test = X_test / 255.0

print("x_train shape: ",X_train.shape)
print("test shape: ",X_test)

# Reshape yapıyoruz 28x28 lik algıladığı için formata dönüştürüyoruz
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("x_test shape: ",X_test.shape)

# Label Encoding -etiketleri işliyoruz 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)

print("x_train shape",X_train.shape)
print("x_test shape",X_test.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_test.shape)

#
from sklearn.metrics import confusion_matrix
#import itertools

#from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# --- Model Olusturma ---
#conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
model = Sequential()
#
model.add(Conv2D(filters = 32, kernel_size = 3,padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
#model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#
model.add(Conv2D(filters = 32, kernel_size = 3,padding = 'Same', 
                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 24, kernel_size = 3,padding = 'Same', 
                 activation ='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 24, kernel_size = 3,padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
# fully connected
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation = "softmax"))

# optimizer tanımlanıyor
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# model compile ediliyor
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 10  # epoch tanımlandı
batch_size = 128 #batch sizetanımlandı

# data augmentation yapılıyor 
##datagen = ImageDataGenerator(
##        featurewise_center=False,  # set input mean to 0 over the dataset
##        samplewise_center=False,  # set each sample mean to 0
##        featurewise_std_normalization=False,  # divide inputs by std of the dataset
##        samplewise_std_normalization=False,  # divide each input by its std
##        zca_whitening=False,  # dimesion reduction
##        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
##        zoom_range = 0.5, # Randomly zoom image 5%
##        width_shift_range=0.5,  # randomly shift images horizontally 5%
##        height_shift_range=0.5,  # randomly shift images vertically 5%
##        horizontal_flip=False,  # randomly flip images
##        vertical_flip=False)  # randomly flip images
##
##datagen.fit(X_train)

# modeli fit ediyoruz 
history = model.fit(X_train,Y_train, batch_size=batch_size,
                              epochs = epochs, validation_data = (X_train,Y_train))

print('*********************************************************')
score = model.evaluate(X_train, Y_train, verbose=0) #train sonuclarini al
print(' (train) loss: {:.4f}'.format(score[0]))
print(' (train) acc: {:.4f}'.format(score[1]))
print(' Train confusion matrix i açılıyor lütfen bekleyin ... ')
# confusion matrix

from itertools import product

classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

#Create Multiclass Confusion Matrix Train Tablo 

preds = model.predict(X_train)
cm = confusion_matrix(np.argmax(Y_train,axis=1), np.argmax(preds,axis=1))

plt.figure(figsize=(8,8))
plt.imshow(cm,cmap=plt.cm.Greens)
plt.title('Confusion Matrix - (Train Tablo)')
plt.colorbar()
plt.xticks(np.arange(10), classes, rotation=90)
plt.yticks(np.arange(10), classes)

for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
    horizontalalignment="center",
    color="white" if cm[i, j] > 500 else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
###------------------------------------------------------------
print('*********************************************************')
score = model.evaluate(X_test, Y_test, verbose=0) #test dosyasi sonuclari al
print(' (test) loss: {:.4f}'.format(score[0]))
print(' (test) acc: {:.4f}'.format(score[1]))
print(' Test confusion matrix i açılıyor lütfen bekleyin ... ')


#Create Multiclass Confusion Matrix Test Tablo

preds = model.predict(X_test)
cm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(preds,axis=1))

plt.figure(figsize=(8,8))
plt.imshow(cm,cmap=plt.cm.Reds)
plt.title('Confusion Matrix - (Test Tablo)')
plt.colorbar()
plt.xticks(np.arange(10), classes, rotation=90)
plt.yticks(np.arange(10), classes)

for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
    horizontalalignment="center",
    color="white" if cm[i, j] > 500 else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print('--- End ---')
