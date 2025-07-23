import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop

# Load data
data = pd.read_csv('A_Z Handwritten Data.csv')
y = data.values[:, 0]
x = data.values[:, 1:].astype('float32') / 255.0
x = np.reshape(x, (x.shape[0], 28, 28, 1))  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Data augmentation
datagen_train = ImageDataGenerator(
                                   validation_split=0.2,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=False,
                                   fill_mode='nearest')

datagen_test = ImageDataGenerator()

data_train = datagen_train.flow(x_train, y_train, subset='training', 
                                batch_size=64, shuffle=True)
data_valid = datagen_train.flow(x_train, y_train, subset='validation',
                                batch_size=64, shuffle=True)
data_test = datagen_test.flow(x_test, y_test, batch_size=1, shuffle=False)

classes = {i:chr(i+65) for i in range(26)}
print("Classes:", classes)

# Model architecture
model = Sequential([
                Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D(2,2),
                Conv2D(32, (3,3), activation='relu'),
                MaxPooling2D(2,2), 
                Flatten(), 
                Dense(512, activation='relu'), 
                Dense(26, activation='softmax')])

# Compile 
model.compile(optimizer=RMSprop(learning_rate=2e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train 
history = model.fit(data_train, 
                    epochs=15,
                    validation_data=data_valid,
                    steps_per_epoch=500,
                    validation_steps=50,
                    verbose=2)

# Evaluate the model
eval_model = model.evaluate(data_test, return_dict=True)
print(f"Test accuracy: {eval_model['accuracy']*100:.2f}%")
model.save('saved_model')