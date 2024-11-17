import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('best_model.keras')

IMGsize = (500,500)
test_images = [
    ('C:\\Users\\YoYo\\OneDrive\\Desktop\\AER850\\Project 2 Data\\Data\\test\\crack\\test_crack.jpg', 'Crack'),
    ('C:\\Users\\YoYo\\OneDrive\\Desktop\\AER850\\Project 2 Data\\Data\\test\\missing-head\\test_missinghead.jpg', 'Missing Head'),
    ('C:\\Users\\YoYo\\OneDrive\\Desktop\\AER850\\Project 2 Data\\Data\\test\\paint-off\\test_paintoff.jpg', 'Paint Off')]

class_labels = {0: 'Crack', 1: 'Missing Head', 2: 'Paint Off'}
def proccess_predict(img_path, model):
    
    img = image.load_img(img_path, target_size=IMGsize)
    
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    predicted_label = class_labels[class_idx]
    
    conf = "\n".join([f"{label}: {pred * 100:.1f}%" for label, pred in zip(class_labels.values(), prediction)])
    
        
    return predicted_label, prediction, conf, img

plt.figure(figsize=(10,5))

for i, (img_path, true_label) in enumerate(test_images):
    predicted_label, prediction, conf, img = proccess_predict(img_path, model)
    
    plt.subplot(1, 3, i+1)
    plt.imshow(img)
    plt.axis('off')
    
    plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
    
    plt.text(10, 20, conf, color='green', fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
    
plt.tight_layout()
plt.show()
    