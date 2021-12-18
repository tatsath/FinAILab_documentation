.. _ann:


ANN-Based Models
----------------

   We have covered the basics of ANNs, along with the architecture of ANNs and their
   training and implementation in Python. Here are are a few additional
   details from the supervised learning perspective

   Neural networks are reducible to a classification or regression model
   with the activation function of the node in the output layer. In
   the case of a regression problem, the output node has linear
   activation function (or no activation function). A linear function
   produces a continuous output ranging from -inf to +inf. Hence, the
   output layer will be the linear function of the nodes in the layer
   before the output layer, and it will be a regression-based model.

   In the case of a classification problem, the output node has a
   sigmoid or softmax activation function. A sigmoid or softmax
   function produces an output ranging from zero to one to represent the
   probability of target value. Softmax function can also be used for
   multiple groups for classification.

ANN using sklearn
~~~~~~~~~~~~~~~~~

   ANN regression and classification models can be constructed using the
   sklearn package of Python, as shown in the following code snippet:

   Classification

   .. code-block:: python
         
      from sklearn.neural_network import MLPClassifier 
      model = MLPClassifier()
      model.fit(X, Y)

   Regression

   .. code-block:: python
      
      from sklearn.neural_network import MLPRegressor 
      model = MLPRegressor()
      model.fit(X, Y)

.. _hyperparameters-5:

Hyperparameters
~~~~~~~~~~~~~~~

   As we saw previously,
   ANN has many hyperparameters. Some of the hyperparameters that are
   present in the sklearn implementation of ANN and can be tweaked while
   performing the grid search are:

   *Hidden Layers (*\ hidden_layer_sizes *in sklearn)*

   It represents the number of layers and nodes in the ANN architecture.
   In sklearn implementation of ANN, the ith element represents the
   number of neurons in the ith hidden layer. A sample value for grid
   search in the sklearn implementation can be [(*20*,), (*50*,),
   (*20*, *20*), (*20*, *30*, *20*)].

   *Activation Function (*\ activation *in sklearn)*

   It represents the activation function of a hidden layer. Some of the
   activation functions defined before, such as sigmoid, relu,
   or tanh, can be used.

Deep neural network
~~~~~~~~~~~~~~~~~~~

   ANNs with more than a single hidden layer are often called deep
   networks. We prefer using the library Keras to implement such
   networks, given the flexibility of the library. The detailed
   implementation of a deep neural network in Keras was shown before. 
   Similar to
   MLPClassifier and MLPRegressor in sklearn for classification and
   regression, Keras has modules called KerasClassifier and
   KerasRegressor that can be used for creating classification and
   regression models with deep network.

   A popular problem in finance is time series prediction, which is
   predicting the next value of a time series based on a historical
   overview. Some of the deep neural networks, such as recurrent
   neural network (RNN), can be directly used for time series
   prediction. The details of this approach are provided in rnn section.

.. _advantages-and-disadvantages-6:

Advantages and disadvantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The main advantage of an ANN is that it captures the nonlinear
   relationship between the variables quite well. ANN can more easily
   learn rich representations and is good with a large number of input
   features with a large dataset. ANN is flexible in how it can be used.
   This is evident from its use across a wide variety of areas in
   machine learning and AI, including reinforcement learning and NLP, as
   discussed before.

   The main disadvantage of ANN is the interpretability of the model,
   which is a drawback that often cannot be ignored and is sometimes
   the determining factor when choosing a model. ANN is not good with
   small datasets and requires a lot of tweaking and guesswork. Choosing
   the right topology/algorithms to solve a problem is difficult.
   Also, ANN is computationally expensive and can take a lot of time to
   train.

Using ANNs for supervised learning in finance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   If a simple model such as linear or logistic regression perfectly
   fits your problem, don’t bother with ANN. However, if you are
   modeling a complex dataset and feel a need for better prediction
   power, give ANN a try. ANN is one of the most flexible models in
   adapting itself to the shape of the data, and using it for supervised
   learning problems can be an interesting and valuable exercise.