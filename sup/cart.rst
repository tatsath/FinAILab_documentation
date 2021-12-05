.. _cart:

Classification and Regression Trees
-----------------------------------

   In the most general terms, the purpose of an analysis via
   tree-building algorithms is to determine a set of *if–then* logical
   (split) conditions that permit accurate prediction or classification
   of cases. *Classification and regression trees* (or *CART* or
   *decision tree classifiers*) are attractive models if we care about
   interpretability. We can think of this model as breaking down our
   data and making a decision based on asking a series of questions.
   This algorithm is the foundation of ensemble methods such as random
   forest and gradient boosting method.

Representation
~~~~~~~~~~~~~~

   The model can be represented by a *binary tree* (or *decision tree*),
   where each node is an input variable *x* with a split point and each
   leaf contains an output variable *y* for prediction.

   |image24|\ `Figure 4-4 <#_bookmark238>`__ shows an example of a
   simple classification tree to predict whether a per‐ son is a male or
   a female based on two inputs of height (in centimeters) and weight
   (in kilograms).

   *Figure 4-4. Classification and regression tree example*

Learning a CART model
~~~~~~~~~~~~~~~~~~~~~

   Creating a binary tree is actually a process of dividing up the input
   space. A *greedy approach* called *recursive binary splitting* is
   used to divide the space. This is a numeri‐ cal procedure in which
   all the values are lined up and different split points are tried and
   tested using a cost (loss) function. The split with the best cost
   (lowest cost, because we minimize cost) is selected. All input
   variables and all possible split points

   are evaluated and chosen in a greedy manner (e.g., the very best
   split point is chosen each time).

   For regression predictive modeling problems, the cost function that
   is minimized to choose split points is the *sum of squared errors*
   across all training samples that fall within the rectangle:

   ∑\ *n* (*y* – *prediction* )2

   where *y\ i* is the output for the training sample and prediction is
   the predicted output for the rectangle. For classification, the *Gini
   cost function* is used; it provides an indi‐ cation of how pure the
   leaf nodes are (i.e., how mixed the training data assigned to each
   node is) and is defined as:

   *G* = ∑\ *n p* \* (1 – *p* )

   where *G* is the Gini cost over all classes and *p\ k* is the number
   of training instances with class *k* in the rectangle of interest. A
   node that has all classes of the same type (perfect class purity)
   will have *G = 0*, while a node that has a *50–50* split of classes
   for a binary classification problem (worst purity) will have *G =
   0.5*.

Stopping criterion
~~~~~~~~~~~~~~~~~~

   The recursive binary splitting procedure described in the preceding
   section needs to know when to stop splitting as it works its way down
   the tree with the training data. The most common stopping procedure
   is to use a minimum count on the number of training instances
   assigned to each leaf node. If the count is less than some minimum,
   then the split is not accepted and the node is taken as a final leaf
   node.

Pruning the tree
~~~~~~~~~~~~~~~~

   The stopping criterion is important as it strongly influences the
   performance of the tree. Pruning can be used after learning the tree
   to further lift performance. The com‐ plexity of a decision tree is
   defined as the number of splits in the tree. Simpler trees are
   preferred as they are faster to run and easy to understand, consume
   less memory during processing and storage, and are less likely to
   overfit the data. The fastest and simplest pruning method is to work
   through each leaf node in the tree and evaluate the effect of
   removing it using a test set. A leaf node is removed only if doing so
   results in a drop in the overall cost function on the entire test
   set. The removal of nodes can be stopped when no further improvements
   can be made.

.. _implementation-in-python-1:

Implementation in Python
~~~~~~~~~~~~~~~~~~~~~~~~

   CART regression and classification models can be constructed using
   the sklearn package of Python, as shown in the following code
   snippet:

   Classification

   from sklearn.tree import DecisionTreeClassifier model =
   DecisionTreeClassifier()

   model.fit(X, Y)

   Regression

   from sklearn.tree import DecisionTreeRegressor model =
   DecisionTreeRegressor ()

   model.fit(X, Y)

.. _hyperparameters-4:

Hyperparameters
~~~~~~~~~~~~~~~

   CART has many hyperparameters. However, the key hyperparameter is the
   maxi‐ mum depth of the tree model, which is the number of components
   for dimensional‐ ity reduction, and which is represented by max_depth
   in the sklearn package. Good values can range from *2* to *30*
   depending on the number of features in the data.

.. _advantages-and-disadvantages-5:

Advantages and disadvantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   In terms of advantages, CART is easy to interpret and can adapt to
   learn complex relationships. It requires little data preparation, and
   data typically does not need to be scaled. Feature importance is
   built in due to the way decision nodes are built. It per‐ forms well
   on large datasets. It works for both regression and classification
   problems.

   In terms of disadvantages, CART is prone to overfitting unless
   pruning is used. It can be very nonrobust, meaning that small changes
   in the training dataset can lead to quite major differences in the
   hypothesis function that gets learned. CART generally has worse
   performance than ensemble models, which are covered next.