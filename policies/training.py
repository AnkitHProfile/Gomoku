import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from google.colab import drive
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime

drive.mount('/content/drive')

w, h = 15, 15
base_path = os.path.join('/content/drive/MyDrive/gomocup20results', '*.npz')

file_list = glob(base_path)

xD, yD = [], []
for file_path in tqdm(file_list):
    data = np.load(file_path)
    xD.extend(data['inputs'])
    yD.extend(data['outputs'])

xD  = np.array(xD, np.float32).reshape((-1, h, w, 1))
yD = np.array(yD, np.float32).reshape((-1, h * w))

x_train, x_val, y_train, y_val = train_test_split(xD, yD, test_size=0.3, random_state=2020)

del xD, yD

print("Training : " + x_train.shape, y_train.shape)
print("Validation : " + x_val.shape, y_val.shape)
#  2D Neural Network
model = models.Sequential([
    layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=(h, w, 1)),
    layers.Conv2D(64, 7, activation='relu', padding='same'),
    layers.Conv2D(128, 7, activation='relu', padding='same'),
    layers.Conv2D(64, 7, activation='relu', padding='same'),
    layers.Conv2D(64, 7, activation='relu', padding='same'),
    layers.Conv2D(1, 1, activation=None, padding='same'),
    layers.Reshape((h * w,)),
    layers.Activation('sigmoid')
])

model.compile(
    optimizer='adam',
    metrics=['acc']
)

model.summary()

name = datetime.now().strftime('%d_%M%S')
os.makedirs('models', exist_ok=True)

model_obj = model.fit(
    x=x_train,
    y=y_train,
    batch_size=256,
    epochs=10,
    callbacks=[
        ModelCheckpoint('./models/%s.h5' % (name), monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto')
    ],
    validation_data=(x_val, y_val),
    use_multiprocessing=True,
    workers=20
)