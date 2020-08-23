import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

'''
Dataset:
Example data:

An experimental drug was tested on individuals from ages 13 to 100.
The trial had 2100 participants. Half were under 65 years old, half were over 65 years old.
95% of patients 65 or older experienced side effects.
95% of patients under 65 experienced no side effects.
'''

# Setting & splitting Train Dataset: train-labels and train-samples
train_labels = []
train_samples = []

# TODO: TRAIN-DATA GENERATION
# Generating 2100 participants
# 1st for loop 50 young + 50 old
# 2nd for loop 1000 young(13,64) + 1000 old(65,100) => TOTAL=2100
for i in range(50):
    # ~5% of younger individuals with side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    # Labeling 1 for side effects individuals
    train_labels.append(1)

    # ~5% of younger individulals with no side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # ~95%  younger individuals with side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # ~95 older with no side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

# ---------------------------------------------------------

# TODO: Preprocessing data for training
# Converting to numpy array convention
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)

# shuffling the both train data for removing imposed order
train_labels, train_samples = shuffle(train_labels, train_samples)

# Normalizing dataset -> to speedup the training of Neural network process
# Rescaling the data from (13,100) t0 (0,1)
# reshape(-1,1) for fit_transform as it need 1 arg + dataset is 1-dimension
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
