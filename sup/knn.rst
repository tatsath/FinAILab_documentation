.. _knn:

K-Nearest Neighbors
-------------------

   *K-nearest neighbors* (KNN) is considered a “lazy learner,” as there
   is no learning required in the model. For a new data point,
   predictions are made by searching through the entire training set for
   the *K* most similar instances (the neighbors) and summarizing the
   output variable for those *K* instances.

   To determine which of the *K* instances in the training dataset are
   most similar to a new input, a distance measure is used. The most
   popular distance measure is *Eucli*

   *dean distance*, which is calculated as the square root of the sum of
   the squared differences between a point *a* and a point *b* across
   all input attributes *i*, and which is represented as *d* (*a*, *b*)
   = ∑\ *n* (*a* – *b* )2. Euclidean distance is a good distance measure to use if the input variables are similar in type.

   Another distance metric is *Manhattan distance*, in which the
   distance between point

   *a* and point *b* is represented as *d* (*a*, *b*) = ∑\ *n* \| *a* –
   *b* \| . Manhattan distance is a good

   measure to use if the input variables are not similar in type. The
   steps of KNN can be summarized as follows:

1. Choose the number of *K* and a distance metric.

2. Find the *K*-nearest neighbors of the sample that we want to
   classify.

3. Assign the class label by majority vote.

..

   KNN regression and classification models can be constructed using the
   sklearn package of Python, as shown in the following code:

   Classification

   .. code-block:: python
      
      from sklearn.neighbors import KNeighborsClassifier 
      model = KNeighborsClassifier()
      model.fit(X, Y)

   Regression

   .. code-block:: python
      
      from sklearn.neighbors import KNeighborsRegressor 
      model = KNeighborsRegressor()
      model.fit(X, Y)

.. _hyperparameters-3:

Hyperparameters
~~~~~~~~~~~~~~~

   The following key parameters are present in the sklearn
   implementation of KNN and can be tweaked while performing the grid
   search:

   *Number of neighbors (*\ n_neighbors *in sklearn)*

   The most important hyperparameter for KNN is the number of neighbors
   (n_neighbors). Good values are between 1 and 20.

   *Distance metric (*\ metric *in sklearn)*

   It may also be interesting to test different distance metrics for
   choosing the com‐ position of the neighborhood. Good values are
   *euclidean* and *manhattan*.

.. _advantages-and-disadvantages-3:

Advantages and disadvantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   In terms of advantages, no training is involved and hence there is no
   learning phase. Since the algorithm requires no training before
   making predictions, new data can be added seamlessly without
   impacting the accuracy of the algorithm. It is intuitive and easy to understand. The model naturally handles multiclass
   classification and can learn complex decision boundaries. KNN is
   effective if the training data is large. It is also robust to noisy
   data, and there is no need to filter the outliers.

   In terms of the disadvantages, the distance metric to choose is not
   obvious and diffi‐ cult to justify in many cases. KNN performs poorly
   on high dimensional datasets. It is expensive and slow to predict new
   instances because the distance to all neighbors must be recalculated.
   KNN is sensitive to noise in the dataset. We need to manually input
   missing values and remove outliers. Also, feature scaling
   (standardization and normalization) is required before applying the
   KNN algorithm to any dataset; other‐ wise, KNN may generate wrong
   predictions.