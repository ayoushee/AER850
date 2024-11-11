#Project 2
import tensorflow as tf
print(tf.__version__)
## Step 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMGsize = (500, 500)
batch = 4

train = 'D:/Project 2 Data/Data/train'
validation = 'D:/Project 2 Data/Data/valid'
test = 'D:/Project 2 Data/Data/test'

# Data Augmentation
train_data = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.25, 
    zoom_range = 0.15, 
    horizontal_flip = True
    )
val_rescale = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

#Train and validation generator (Keras)
train_gen = train_data.flow_from_directory(
    train, 
    target_size = IMGsize, 
    batch_size = batch, 
    class_mode = 'categorical'
    )
validation_gen = val_rescale.flow_from_directory(
    validation, 
    target_size = IMGsize, 
    batch_size = batch, 
    class_mode = 'categorical'
    )
test_gen = test_data.flow_from_directory(
    test, 
    target_size = IMGsize, 
    batch_size = batch, 
    class_mode = 'categorical'
    )

## Step 2
import tensorflow as tf
from tensorflow.keras import models 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, LeakyReLU, ReLU

# Convolutional Layer & MaxPooling
model = models.Sequential()
model.add(Conv2D(32, (3,3), activation = None, input_shape=(500,500,3)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation = None))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation = None))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))

tf.config.list_physical_devices('GPU')

# Flatten
model.add(Flatten())

# Dense and Dropout
model.add(Dense(128, activation = 'relu'))
model.add(ReLU())
model.add(Dense(3, activation='softmax'))

## Step 3 will come backj to this after i have evaluated the model in step 4

model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

## Step 4
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystop = EarlyStopping(monitor='val_loss', patience=5)
check = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(
    train_gen, 
    validation_data=validation_gen, 
    epochs=10, 
    verbose=1, 
    callbacks=[check, earlystop])

import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()