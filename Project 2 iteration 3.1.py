#Project 2
import tensorflow as tf
print(tf.__version__)
## Step 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMGsize = (500, 500)
batch = 32

train = './Project 2 Data/Data/train'
validation = './Project 2 Data/Data/valid'
test = './Project 2 Data/Data/test'

# Data Augmentation
train_data = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.3, 
    zoom_range = 0.2, 
    horizontal_flip = True)
val_rescale = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

#Train and validation generator (Keras)
train_gen = train_data.flow_from_directory(train, target_size = IMGsize, batch_size = batch, class_mode = 'categorical')
validation_gen = val_rescale.flow_from_directory(validation, target_size = IMGsize, batch_size = batch, class_mode = 'categorical')
test_gen = test_data.flow_from_directory(test, target_size = IMGsize, batch_size = batch, class_mode = 'categorical')

## Step 2
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Convolutional Layer & MaxPooling
model = tf.keras.Sequential([
    Conv2D(32, (3,3), (1,1), activation = 'relu', input_shape=(500,500,3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), (1,1), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
   
    Conv2D(128, (3,3), (1,1), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
 
    Conv2D(256, (3,3), (1,1), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
 
    #flatten
    Flatten(),
     
    # Dense and Dropout
    Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.02)),
    Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.6),
    Dense(3, activation='softmax')
    ])

## Step 3 will come backj to this after i have evaluated the model in step 4
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer= optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

## Step 4
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystop = EarlyStopping(monitor = 'val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_v3.keras', monitor='val_loss', save_best_only=True)
history = model.fit(train_gen, validation_data=validation_gen, epochs=50, batch_size= batch, callbacks=[checkpoint, earlystop])

import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()

plt.show()