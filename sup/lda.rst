.. _lda:

Linear Discriminant Analysis
----------------------------

   The objective of the *linear discriminant analysis* (LDA) algorithm
   is to project the data onto a lower-dimensional space in a way that
   the class separability is maximized and the variance within a class
   is minimized.

   During the training of the LDA model, the statistical properties
   (i.e., mean and cova‐ riance matrix) of each class are computed. The
   statistical properties are estimated on the basis of the following
   assumptions about the data:

-  Data is `normally distributed <https://oreil.ly/cuc7p>`__, so that
   each variable is shaped like a bell curve when plotted.

-  Each attribute has the same variance, and the values of each variable
   vary around the mean by the same amount on average.

..

   To make a prediction, LDA estimates the probability that a new set of
   inputs belongs to every class. The output class is the one that has
   the highest probability.

Implementation in Python and hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The LDA classification model can be constructed using the sklearn
   package of Python, as shown in the following code snippet:

   .. code-block:: python
   
      from sklearn.discriminant_analysis 
      import LinearDiscriminantAnalysis
      model = LinearDiscriminantAnalysis()
      model.fit(X, Y)

   The key hyperparameter for the LDA model is number of components for
   dimensionality reduction, which is represented by n_components in
   sklearn.

Advantages and disadvantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   In terms of advantages, LDA is a relatively simple model with fast
   implementation and is easy to implement. In terms of disadvantages,
   it requires feature scaling and involves complex matrix operations.