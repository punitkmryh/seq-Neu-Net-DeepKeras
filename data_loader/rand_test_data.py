import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Setting & splitting Train Dataset: train-labels and train-samples
test_samples = []
test_labels = []

# TODO: TEST-DATA GENERATION for 100% participants
print('----------------------------------------------------------------------')
print('[INFO] Generating & Splitting train dataset using randint function....')
print('----------------------------------------------------------------------')
for i in range(50):
    # ~5% of younger individuals with side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # ~5% of younger individulals with no side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    # ~95%  younger individuals with side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # ~95 older with no side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

# TODO: Preprocessing data for training
# Converting to numpy array convention
test_samples = np.array(test_samples)
test_labels = np.array(test_labels)

# shuffling the both train data for removing imposed order
test_labels, test_samples = shuffle(test_labels, test_samples)

# Normalizing dataset -> to speedup the training of Neural network process
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

