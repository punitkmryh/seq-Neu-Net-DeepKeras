# **Keras Deep Learning**

Keras provides a complete framework to create any type of neural networks. Keras is innovative as well as very easy to learn. It supports simple neural network to very large and complex neural network model. Let us understand the architecture of Keras framework and how Keras helps in deep learning in this chapter.

**Architecture of Keras**

Keras API can be divided into three main categories

- Core 
- Module
- Model



In Keras, every ANN is represented by **Keras Models**. In turn, every Keras Model is composition of **Keras Layers** and represents ANN layers like input, hidden layer, output layers, convolution layer, pooling layer, etc., Keras model and layer access **Keras modules** for activation function, loss function, regularization function, etc., Using Keras model, Keras Layer, and Keras modules, any ANN algorithm (CNN, RNN, etc.,) can be represented in a simple and efficient manner.

#### **Architecture of Keras**![Architecture of keras](https://github.com/punitkmryh/seq-Neu-Net-DeepKeras/blob/master/img/Architecture%20of%20keras.png)

**Model**

Keras Models are of two types as mentioned below



- **Sequential Model** − Sequential model is basically **simplest** type of model built using keras or tensorflow, it is defined as a **liner stack of model** (a linear composition of Keras Layers). Sequential model is easy, minimal as well as has the ability to represent nearly all available neural networks.Keras Models are of two types as mentioned below

  ```python
  from keras.models import Sequential 
  from keras.layers import Dense, Activation 
  
  model = Sequential()  
  model.add(Dense(512, activation = 'relu', input_shape = (784,)))
  ```

  **Layer**

  Each Keras layer in the Keras model represent the corresponding layer (input layer, hidden layer and output layer) in the actual proposed neural network model. Keras provides a lot of pre-build layers so that any complex neural network can be easily created.

  #### **WorkFlow of ANN using Keras** ![Workflow of ANN-2](https://github.com/punitkmryh/seq-Neu-Net-DeepKeras/blob/master/img/Workflow%20of%20ANN-2.png)

**Core Modules**

Keras also provides a lot of built-in neural network related functions to properly create the Keras model and Keras layers.

- **Activations module** − Activation function is an important concept in ANN and activation modules provides many activation function like softmax, relu, etc.
- **Loss module** − Loss module provides loss functions like mean_squared_error, mean_absolute_error, poisson, etc.
- **Optimizer module** − Optimizer module provides optimizer function like adam, sgd, etc.,
- **Regularizers** − Regularizer module provides functions like L1 regularizer, L2 regularizer, etc.

## **The Sequential model:**

### Introduction:

**When to use a Sequential model**

A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

Schematically, the following Sequential model:

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
 # Call model on a test input
x = tf.ones((3, 3))
y = model(x)
```

is equivalent to this function:

```python
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name=“layer3") 

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

A Sequential model is **not appropriate** when:

- Your model has multiple inputs or multiple outputs
- Any of your layers has multiple inputs or multiple outputs
- You need to do layer sharing
- You want non-linear topology (e.g. a residual connection, a multi-branch model)



You can also create a Sequential model incrementally via the `add()` method:

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

Note that there's also a corresponding pop() method to remove layers: a Sequential model behaves very much like a list of layers.

```python
model.pop()
print(len(model.layers))  # 2
```

Also note that the Sequential constructor accepts a name argument, just like any layer or model in Keras. This is useful to annotate TensorBoard graphs with semantically meaningful names.

```python
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
```



### TODO: Building a NeuNet model

```python
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
```

#### <u>Dataset</u>:

**Generating Example data:**

- *An experiemental drug was tested on individuals from ages 13 to 100.*
- *The trial had 2100 participants. Half were under 65 years old, half were over 65 years old.*
- *95% of patientes 65 or older experienced side effects.*
- *95% of patients under 65 experienced no side effects.*

```python
# Setting train label and sample
train_lables=[]
train_samples=[]
```

```python
# TODO: DATA GENERATION
# Generating and splitting Dataset:
for i in range(50):
    #     ~5% of younger individuals with side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    # Labeling 1 for side effects individuals
    train_lables.append(1)

    #     ~5% of younger individulals with no side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_lables.append(0)
```

```python
for i in range(1000):
    # ~95%  younger individuals with side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_lables.append(1)

    # ~95 older with no side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_lables.append(0)
```

#### <u>Preprocessing</u>

```python
# TODO: Preprocessing data for training
# Converting to numpy array convention
train_samples=np.array(train_samples)
train_lables=np.array(train_lables)

# shuffling the both train data for removing imposed order
train_samples,train_lables=shuffle(train_samples, train_lables)

# Normalizing dataset -> to speedup the training of Neural network process
# Rescaling the data from (13,100) t0 (0,1)
# reshape(-1,1) for fit_transform as it need 1 arg + dataset is 1-dimension
scaler = MinMaxScaler(feature_range=(0,1))
scaled_trained_samples = scaler.fit_transform(train_samples.reshape(-1,1))
```

#### <u>NeuNet Building</u>

```markdown
TODO: Building, Training and Validating Model

* Building NeuNet Model
    * Visualizing model
* Compiling and Fitting model
* Objective: compiling and fitting model with validation
```

- **`Dense`** - A dense layer is just a regular layer of neurons in a neural network. 
  - Each neuron recieves input from all the neurons in the previous layer, thus densely connected.
  - `ReLU` - Rectified Linear Unit - `y = max(0,x)`
  - ReLU linear function that will output the input directly if it is positive, otherwise, it will output zero.
    - responsible for transforming the summed weighted input from the node 
        into the activation of the node or output for that input.

- **`Softmax`** - gives output probability for each output class
     - used only in the output layer of a neural net to represent a probability distribution of possible outcomes of the network.
     - when we want to build a multi-class classifier which solves the problem of assigning an instance to one class when the number of possible classes is larger than two.

```python
model = Sequential([
    # input_shape - automatically determines input for model
    Dense(name="in_next", units=16, input_shape=(1,), activation='relu'),
    Dense(name='layer2', units=32, activation='relu'),
    # layer3 gives output label
    Dense(name="next_out", units=2, activation='softmax')
])

# Gives Visual represent of built model
model.summary()
```

Result:

```bash
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
```

#### Preparing model for Training

- **`Adam optimizer`** - algorithm that can be used instead of the classical stochastic gradient descent procedure to `update network weights iterative` based in training data.

- **`Sparse_categorical_crossentropy`** - Also called Softmax Loss, its Softmax activation plus a Cross-Entropy loss
     - `sparse_categorical_crossentropy` - when your classes are mutually exclusive (e.g. when each sample belongs exactly to one class) 
          ategorical_crossentropy` - when one sample can have multiple classes or labels are soft probabilities (like [0.5, 0.3, 0.2]).

```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- **`batch_size`** - divide dataset into Number of Batches or sets or parts

- **`epochs`** - One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.

  - One epoch leads to under-fitting of the curve in the graph. As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from `under-fitting` to optimal to `over-fitting curve`.

    

![epoch](/Users/nitishharsoor/Prop/epoch.png)

```python
# `shuffle` is true default and important @ validation data
#  `verbose` - is visual represent in terminal recommended `verbose=2`.
#  ( 0=silent, 1=progress bar, 2=one line per epochs)
model.fit(x=rand_dataset.scaled_trained_samples, y=rand_dataset.train_lables, batch_size=10, epochs=30, shuffle=True,
          verbose=2)
```

Result of model on compiling and fitting

```bash
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
```

#### Building validation set with keras

##### What Is A Validation Set?

- **`validation set`** -  its new data applied on trained model, and have the model accurately predict on data that 
  it hasn’t seen before based solely on what it learned from the training set.

 > A **validation dataset** is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model’s hyperparameters.

- Before training begins, we can choose to remove a portion of the training set and place it in a validation set. Then, during training, the model will train *only* on the training set, and it will validate by evaluating the data in the validation set.

- Essentially, the model is learning the features of the data in the training set, taking what it's learned from this data, and then predicting on the validation set. During each epoch, we will see not only the loss and accuracy results for the training set, but also for the validation set.

  - This allows us to see how well the model is generalizing on data it wasn’t trained on because, recall, the validation data should *not* be part of the training data.

  - This also helps us see whether or not the model is **overfitting**. Overfitting occurs when the model only learns the specifics of the training data and is unable to generalize well on data that it wasn’t trained on.

    
  
    > *Suppose that we would like to **estimate the test error associated with fitting** a particular statistical learning method on a set of observations. The validation set approach […] is a very simple strategy for this task. It involves randomly dividing the available set of observations into two parts, a training set and a validation set or hold-out set. The model is fit on the training set, and the fitted model is used to predict the responses for the observations in the validation set. The resulting validation set error rate — typically assessed using MSE in the case of a quantitative response—provides an estimate of the test error rate.*

##### Definitions of Train, Validation, and Test Datasets

- **`Training Dataset`**: The sample of data used to fit the model.
- **`Validation Dataset`**: The sample of data used to provide an <u>unbiased evaluation of a model fit</u> **on the training dataset while tuning model hyperparameters**. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.
- **`Test Dataset`**: The sample of data used to provide an unbiased evaluation of a **final model fit** <u>on the training dataset</u>.

```python
# split data
data = ...
train, validation, test = split(data)
 
# tune model hyperparameters during training
parameters = ...
for params in parameters:
	model = fit(train, params)
	skill = evaluate(model, validation)
 
# evaluate final model for comparison with other models
model = fit(train)
skill = evaluate(model, test)
```

Below are some additional clarifying notes:

- The validation dataset may also play a role in other forms of model preparation, such as **feature selection.**
- The final model could be fit on the aggregate of the training and validation datasets.

