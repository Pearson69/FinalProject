# AlphaRec
A convolutional neural network (CNN) implementation using TensorFlow/Keras to recognize handwritten Latin characters (A-Z).

![MNIST Database](https://github.com/Pearson69/FinalProject/blob/54c29da6616730666b3262d8b25be09201645888/English_MNIST.png)

## Dataset 
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

## Software
TensorFlow 2.x with XLA acceleration

Python 3.9+

NVIDIA CUDA-enabled drivers

## Model Architercture 

1. **Input Layer**
   - Input shape: `(28, 28, 1)` (grayscale images of 28×28 pixels)


2. **First Convolutional Block**
   - `Conv2D(32, (3,3), activation='relu')`
     - 32 filters with 3×3 kernel size
     - ReLU activation function
     - Output shape: `(26, 26, 32)
   - `MaxPooling2D(2,2)`
     - 2×2 max pooling
     - Output shape: `(13, 13, 32)`

3. **Second Convolutional Block**
   - `Conv2D(32, (3,3), activation='relu')`
     - 32 filters with 3×3 kernel size
     - ReLU activation
     - Output shape: `(11, 11, 32)`
   - `MaxPooling2D(2,2)`
     - 2×2 max pooling
     - Output shape: `(5, 5, 32)` (rounded down from 5.5)

4. **Flatten Layer**
   - `Flatten()`
     - Converts 3D feature maps to 1D vector
     - Output shape: `(800,)` (5 × 5 × 32 = 800)

5. **First Dense Layer**
   - `Dense(512, activation='relu')`
     - Fully connected layer with 512 neurons
     - ReLU activation

6. **Output Layer**
   - `Dense(26, activation='softmax')`
     - 26 neurons 
     - Softmax activation 

## Final Results

**99.08%** Test Accuracy  
**97.35%** Validation Accuracy  
**0.0324** Test Loss  

## Run This Project
1. Install all the dependencies
```bash
  pip install -r requirements.txt
```
*if you are running this on jetson orin, make sure you configure tensorflow correctly [see this guide from Nvidia](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)

2. Download the dataset by running download.py

3. Run train.py to train the model
   
*make sure to have correct path configured for the dataset 

4. Run example.py to see a simple flask-based web implementation of the trained model 
