.. _ensemble:

Ensemble Models
---------------

   The goal of *ensemble models* is to combine different classifiers
   into a meta-classifier that has better generalization performance
   than each individual classifier alone. For example, assuming that we
   collected predictions from 10 experts, ensemble methods would allow
   us to strategically combine their predictions to come up with a
   predic‐ tion that is more accurate and robust than the experts’
   individual predictions.

   The two most popular ensemble methods are bagging and boosting.
   *Bagging* (or *boot‐ strap aggregation*) is an ensemble technique of
   training several individual models in a parallel way. Each model is
   trained by a random subset of the data. *Boosting*, on the other
   hand, is an ensemble technique of training several individual models
   in a sequential way. This is done by building a model from the
   training data and then

   creating a second model that attempts to correct the errors of the
   first model. Models are added until the training set is predicted
   perfectly or a maximum number of mod‐ els is added. Each individual
   model learns from mistakes made by the previous model. Just like the
   decision trees themselves, bagging and boosting can be used for
   classification and regression problems.

   By combining individual models, the ensemble model tends to be more
   flexible (less bias) and less data-sensitive (less
   variance).\ `5 <#_bookmark253>`__ Ensemble methods combine multiple,
   simpler algorithms to obtain better performance.

   In this section we will cover random forest, AdaBoost, the gradient
   boosting method, and extra trees, along with their implementation
   using sklearn package.

   **Random forest.** *Random forest* is a tweaked version of bagged
   decision trees. In order to understand a random forest algorithm, let
   us first understand the *bagging algo‐ rithm*. Assuming we have a
   dataset of one thousand instances, the steps of bagging are:

4. Create many (e.g., one hundred) random subsamples of our dataset.

5. Train a CART model on each sample.

6. Given a new dataset, calculate the average prediction from each model
   and aggregate the prediction by each tree to assign the final label
   by majority vote.

..

   A problem with decision trees like CART is that they are greedy. They
   choose the variable to split by using a greedy algorithm that
   minimizes error. Even after bagging, the decision trees can have a
   lot of structural similarities and result in high correla‐ tion in
   their predictions. Combining predictions from multiple models in
   ensembles works better if the predictions from the submodels are
   uncorrelated, or at best are weakly correlated. Random forest changes
   the learning algorithm in such a way that the resulting predictions
   from all of the subtrees have less correlation.

   In CART, when selecting a split point, the learning algorithm is
   allowed to look through all variables and all variable values in
   order to select the most optimal split point. The random forest
   algorithm changes this procedure such that each subtree can access
   only a random sample of features when selecting the split points. The
   number of features that can be searched at each split point (*m*)
   must be specified as a parameter to the algorithm.

   As the bagged decision trees are constructed, we can calculate how
   much the error function drops for a variable at each split point. In
   regression problems, this may be the drop in sum squared error, and
   in classification, this might be the Gini cost. The

5. Bias and variance are described in detail later in this chapter.

..

   bagged method can provide feature importance by calculating and
   averaging the error function drop for individual variables.

   **Implementation in Python.** Random forest regression and
   classification models can be constructed using the sklearn package of
   Python, as shown in the following code:

   Classification

   from sklearn.ensemble import RandomForestClassifier model =
   RandomForestClassifier()

   model.fit(X, Y)

   Regression

   from sklearn.ensemble import RandomForestRegressor model =
   RandomForestRegressor()

   model.fit(X, Y)

   **Hyperparameters.** Some of the main hyperparameters that are
   present in the sklearn implementation of random forest and that can
   be tweaked while performing the grid search are:

   *Maximum number of features (*\ max_features *in sklearn)*

   This is the most important parameter. It is the number of random
   features to sample at each split point. You could try a range of
   integer values, such as 1 to 20, or 1 to half the number of input
   features.

   *Number of estimators (*\ n_estimators *in sklearn)*

   This parameter represents the number of trees. Ideally, this should
   be increased until no further improvement is seen in the model. Good
   values might be a log scale from 10 to 1,000.

   **Advantages and disadvantages.** The random forest algorithm (or
   model) has gained huge popularity in ML applications during the last
   decade due to its good perfor‐ mance, scalability, and ease of use.
   It is flexible and naturally assigns feature impor‐ tance scores, so
   it can handle redundant feature columns. It scales to large datasets
   and is generally robust to overfitting. The algorithm doesn’t need
   the data to be scaled and can model a nonlinear relationship.

   In terms of disadvantages, random forest can feel like a black box
   approach, as we have very little control over what the model does,
   and the results may be difficult to interpret. Although random forest
   does a good job at classification, it may not be good for regression
   problems, as it does not give a precise continuous nature predic‐
   tion. In the case of regression, it doesn’t predict beyond the range
   in the training data and may overfit datasets that are particularly
   noisy.

Extra trees
~~~~~~~~~~~

   *Extra trees*, otherwise known as *extremely randomized trees*, is a
   variant of a random forest; it builds multiple trees and splits nodes
   using random subsets of features simi‐ lar to random forest. However,
   unlike random forest, where observations are drawn with replacement,
   the observations are drawn without replacement in extra trees. So
   there is no repetition of observations.

   Additionally, random forest selects the best split to convert the
   parent into the two most homogeneous child
   nodes.\ `6 <#_bookmark260>`__ However, extra trees selects a random
   split to divide the parent node into two random child nodes. In extra
   trees, randomness doesn’t come from bootstrapping the data; it comes
   from the random splits of all observations.

   In real-world cases, performance is comparable to an ordinary random
   forest, some‐ times a bit better. The advantages and disadvantages of
   extra trees are similar to those of random forest.

   **Implementation in Python.** Extra trees regression and
   classification models can be con‐ structed using the sklearn package
   of Python, as shown in the following code snippet. The
   hyperparameters of extra trees are similar to random forest, as shown
   in the pre‐ vious section:

   Classification

   from sklearn.ensemble import ExtraTreesClassifier model =
   ExtraTreesClassifier()

   model.fit(X, Y)

   Regression

   from sklearn.ensemble import ExtraTreesRegressor model =
   ExtraTreesRegressor()

   model.fit(X, Y)

Adaptive Boosting (AdaBoost)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   *Adaptive Boosting* or *AdaBoost* is a boosting technique in which
   the basic idea is to try predictors sequentially, and each subsequent
   model attempts to fix the errors of its predecessor. At each
   iteration, the AdaBoost algorithm changes the sample distri‐ bution
   by modifying the weights attached to each of the instances. It
   increases the weights of the wrongly predicted instances and
   decreases the ones of the correctly predicted instances.

6. Split is the process of converting a nonhomogeneous parent node into
   two homogeneous child nodes (best possible).

..

   The steps of the AdaBoost algorithm are:

7.  Initially, all observations are given equal weights.

8.  A model is built on a subset of data, and using this model,
    predictions are made on the whole dataset. Errors are calculated by
    comparing the predictions and actual values.

9.  While creating the next model, higher weights are given to the data
    points that were predicted incorrectly. Weights can be determined
    using the error value. For instance, the higher the error, the more
    weight is assigned to the observation.

10. This process is repeated until the error function does not change,
    or until the maximum limit of the number of estimators is reached.

..

   **Implementation in Python.** AdaBoost regression and classification
   models can be con‐ structed using the sklearn package of Python, as
   shown in the following code snippet:

   Classification

   from sklearn.ensemble import AdaBoostClassifier model =
   AdaBoostClassifier()

   model.fit(X, Y)

   Regression

   from sklearn.ensemble import AdaBoostRegressor model =
   AdaBoostRegressor()

   model.fit(X, Y)

   **Hyperparameters.** Some of the main hyperparameters that are
   present in the sklearn implementation of AdaBoost and that can be
   tweaked while performing the grid search are as follows:

   *Learning rate (*\ learning_rate *in sklearn)*

   Learning rate shrinks the contribution of each classifier/regressor.
   It can be con‐ sidered on a log scale. The sample values for grid
   search can be 0.001, 0.01, and 0.1.

   *Number of estimators (*\ n_estimators *in sklearn)*

   This parameter represents the number of trees. Ideally, this should
   be increased until no further improvement is seen in the model. Good
   values might be a log scale from 10 to 1,000.

   **Advantages and disadvantages.** In terms of advantages, AdaBoost
   has a high degree of precision. AdaBoost can achieve similar results
   to other models with much less tweaking of parameters or settings.
   The algorithm doesn’t need the data to be scaled and can model a
   nonlinear relationship.

   In terms of disadvantages, the training of AdaBoost is time
   consuming. AdaBoost can be sensitive to noisy data and outliers, and
   data imbalance leads to a decrease in clas‐ sification accuracy

Gradient boosting method
~~~~~~~~~~~~~~~~~~~~~~~~

   *Gradient boosting method* (GBM) is another boosting technique
   similar to AdaBoost, where the general idea is to try predictors
   sequentially. Gradient boosting works by sequentially adding the
   previous underfitted predictions to the ensemble, ensuring the errors
   made previously are corrected.

   The following are the steps of the gradient boosting algorithm:

1. A model (which can be referred to as the first weak learner) is built
   on a subset of data. Using this model, predictions are made on the
   whole dataset.

2. Errors are calculated by comparing the predictions and actual values,
   and the loss is calculated using the loss function.

3. A new model is created using the errors of the previous step as the
   target vari‐ able. The objective is to find the best split in the
   data to minimize the error. The predictions made by this new model
   are combined with the predictions of the previous. New errors are
   calculated using this predicted value and actual value.

4. This process is repeated until the error function does not change or
   until the maximum limit of the number of estimators is reached.

..

   Contrary to AdaBoost, which tweaks the instance weights at every
   interaction, this method tries to fit the new predictor to the
   residual errors made by the previous predictor.

   **Implementation in Python and hyperparameters.** Gradient boosting
   method regression and classification models can be constructed using
   the sklearn package of Python, as shown in the following code
   snippet. The hyperparameters of gradient boosting method are similar
   to AdaBoost, as shown in the previous section:

   Classification

   from sklearn.ensemble import GradientBoostingClassifier model =
   GradientBoostingClassifier()

   model.fit(X, Y)

   Regression

   from sklearn.ensemble import GradientBoostingRegressor model =
   GradientBoostingRegressor()

   model.fit(X, Y)

   **Advantages and disadvantages.** In terms of advantages, gradient
   boosting method is robust to missing data, highly correlated
   features, and irrelevant features in the same way as random forest.
   It naturally assigns feature importance scores, with slightly better
   performance than random forest. The algorithm doesn’t need the data
   to be scaled and can model a nonlinear relationship.

   In terms of disadvantages, it may be more prone to overfitting than
   random forest, as the main purpose of the boosting approach is to
   reduce bias and not variance. It has many hyperparameters to tune, so
   model development may not be as fast. Also, fea‐ ture importance may
   not be robust to variation in the training dataset.