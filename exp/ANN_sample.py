from keras.models import Sequential
from keras.layers import Dense, Activation
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

# TODO:Creating a simple ANN model
model = Sequential([
    # 5 Hidden layers + 3 input neurons + relu ``activation``
    Dense(5,input_shape=(3,),activation='relu'),
    # 2 Output layers + softmax ``activation``
    Dense(2,activation='softmax')])

img = np.expand_dims(ndimage.imread("NeuNet.PNG"),0)
plt.imshow(img[0])