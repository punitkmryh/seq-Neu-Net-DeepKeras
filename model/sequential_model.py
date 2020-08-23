import os.path
from data_loader import rand_train_data
from data_loader import rand_test_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# ------------------------------------------------
# TODO: Training ANN
# ------------------------------------------------
'''
    TODO: Building, Training and Validating Model

    * Building NeuNet Model
        * Visualizing model
    * Compiling and Fitting model
    * Objective: compiling and fitting model with validation
'''

# --------------------------------------------------
# TODO: Building a NeuNet model

# Building Hidden layers for NeuNet
'''
    `Dense` - A dense layer is just a regular layer of neurons in a neural network. 
        Each neuron gets input from all the neurons in the previous layer, thus densely connected.
         
    `ReLU` - Rectified Linear Unit - `y = max(0,x)`
    ReLU linear function that will output the input directly if it is positive, otherwise, it will output zero.
    - responsible for transforming the summed weighted input from the node 
        into the activation of the node or output for that input.

    `Softmax` - gives output probability for each output class
        - used only in the output layer of a neural net to represent 
            a probability distribution of possible outcomes of the network.
        - when we want to build a multi-class classifier which solves the problem of assigning an instance to one class 
            when the number of possible classes is larger than two.
'''
# ------------------------------------------------
print('[INFO] Loading datasets form data_loader....')
print('[INFO] Applying MiniMax Scaling......')
print('[INFO] Building simple Sequential NeuNet Model.......')
print('----------------------------------------------------------------------')

NeuNetmodel = Sequential([
    # input_shape - automatically determines input for model
    Dense(name="in_next", units=16, input_shape=(1,), activation='relu'),
    Dense(name='layer2', units=32, activation='relu'),
    # layer3 gives output label
    Dense(name="next_out", units=2, activation='softmax')
])

# Gives Visual represent of built model
NeuNetmodel.summary()

'''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    in_next (Dense)              (None, 16)                32        
    _________________________________________________________________
    layer2 (Dense)               (None, 32)                544       
    _________________________________________________________________
    next_out (Dense)             (None, 2)                 66        
    =================================================================
    Total params: 642
    Trainable params: 642
    Non-trainable params: 0
    _________________________________________________________________
'''

# ------------------------------------------------
# TODO: Preparing model for Training
'''
    `Adam optimizer` - algorithm that can be used instead of the classical stochastic gradient descent procedure
        to `update network weights iterative` based in training data.
        
    `Sparse_categorical_crossentropy` - Also called Softmax Loss, its Softmax activation plus a Cross-Entropy loss
        - `sparse_categorical_crossentropy` - when your classes are mutually exclusive (e.g. when each sample belongs exactly to one class) 
        - `categorical_crossentropy` - when one sample can have multiple classes or labels are soft probabilities (like [0.5, 0.3, 0.2]).
'''
print('-----------------------------------------------------------------------------------------------------')
print('[INFO] Compiling the model before fitting with Adam-optimizer + SCC-loss+ Accuracy-metrics...........')
NeuNetmodel.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model using `fit` function
'''
    `batch_size` - divide dataset into Number of Batches or sets or parts
    
    `epochs` - One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
        One epoch leads to under-fitting of the curve in the graph. As the number of epochs increases, 
        more number of times the weight are changed in the neural network and the curve goes 
        from under-fitting to optimal to over-fitting curve.
'''

# `shuffle` is true default and important @ validation data
#  `verbose` - is visual represent in terminal recommended `verbose=2`.
#  ( 0=silent, 1=progress bar, 2=one line per epochs)
# print('----------------------------------------------------------------------')
# print('[INFO] Fitting model with train scaled data + train label data.........')
# NeuNetmodel.fit(x=rand_train_data.scaled_train_samples, y=rand_train_data.train_labels, batch_size=10, epochs=30,
#                 shuffle=True, verbose=2)
'''
    Epoch 1/30
    210/210 - 0s - loss: 0.6530 - accuracy: 0.5419 👈🏻
    Epoch 2/30
    210/210 - 0s - loss: 0.6142 - accuracy: 0.6776
    ...
    ...
    Epoch 15/30
    210/210 - 0s - loss: 0.1518 - accuracy: 0.9738
    Epoch 16/30
    210/210 - 0s - loss: 0.1412 - accuracy: 0.9738
    ...
    ...
    Epoch 29/30
    210/210 - 0s - loss: 0.0773 - accuracy: 0.9862
 👉🏼 Epoch 30/30
    210/210 - 0s - loss: 0.0748 - accuracy: 0.9862 👈🏻
'''

# -----------------------------------------------------------
# TODO: Building validation set with keras - checking Over-fitting problem
'''
    `validation set` -  its new data not applied during trained model, 
        but for us to see how well the model accurately predict on data that 
        it has not seen before based solely on what it learned from the training set.
'''

# `shuffle` is true default and important @ validation data
#  `verbose` - is visual represent in terminal recommended `verbose=2`.
#  ( 0=silent, 1=progress bar, 2=one line per epochs)
print('------------------------------------------------------------------------------------')
print('[INFO] Fitting model with validation dataset for test error estimation..............')
print('-------------------------------------------------------------------------------------')
NeuNetmodel.fit(x=rand_train_data.scaled_train_samples, y=rand_train_data.train_labels, validation_split=0.1, batch_size=10,
                epochs=30, shuffle=True, verbose=2)
'''
    Epoch 1/30
    189/189 - 1s - loss: 0.0913 - accuracy: 0.9931 - val_loss: 0.1056 - val_accuracy: 0.9571 👈🏻
    Epoch 2/30
    189/189 - 0s - loss: 0.0888 - accuracy: 0.9831 - val_loss: 0.1026 - val_accuracy: 0.9571
    ...
    ...
    Epoch 15/30
    189/189 - 0s - loss: 0.0646 - accuracy: 0.9931 - val_loss: 0.0801 - val_accuracy: 0.9571
    Epoch 16/30
    189/189 - 0s - loss: 0.0639 - accuracy: 0.9889 - val_loss: 0.0779 - val_accuracy: 0.9810
    ...
    ...
    Epoch 29/30
    189/189 - 0s - loss: 0.0516 - accuracy: 0.9937 - val_loss: 0.0661 - val_accuracy: 0.9810
 👉🏼 Epoch 30/30
    189/189 - 0s - loss: 0.0510 - accuracy: 0.9937 - val_loss: 0.0653 - val_accuracy: 0.9810 👈🏻
'''
# TODO: Save and Load a Model with TensorFlow's Keras API
'''
    This save functions saves:
    
    The architecture of the model, allowing to re-create the model.
    The weights of the model.
    The training configuration (loss, optimizer).
    The state of the optimizer, allowing to resume training exactly where you left off.
'''

print('----------------------------------------------------------------------')
print("[INFO] Saving model in to .h5 file.......")
print('----------------------------------------------------------------------')
if os.path.isfile('model_repo/medical_trial_model.h5') is False:
    NeuNetmodel.save('./model_repo/medical_trial_model.h5')


