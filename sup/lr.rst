.. _lr:



Linear Regression (Ordinary Least Squares)
------------------------------------------

   *Linear regression* (Ordinary Least Squares Regression or OLS
   Regression) is perhaps one of the most well-known and best-understood
   algorithms in statistics and machine learning. Linear regression is a
   linear model, e.g., a model that assumes a linear relationship
   between the input variables (*x*) and the single output variable
   (*y*). The goal of linear regression is to train a linear model to
   predict a new *y* given a pre‐ viously unseen *x* with as little
   error as possible.

   Our model will be a function that predicts *y* given *x*\ :sub:`1`,
   *x*\ :sub:`2`...\ *x\ i*:

   *y* = *β*\ :sub:`0` + *β*\ :sub:`1`\ *x*\ :sub:`1` + ... + *β\ i
   x\ i*

   where, *β*\ :sub:`0` is called intercept and *β*\ :sub:`1`...\ *β\ i*
   are the coefficient of the regression.

Implementation in Python
~~~~~~~~~~~~~~~~~~~~~~~~

   from sklearn.linear_model import LinearRegression model =
   LinearRegression()

   model.fit(X, Y)

   In the following section, we cover the training of a linear
   regression model and grid search of the model. However, the overall
   concepts and related approaches are appli‐ cable to all other
   supervised learning models.

Training a model
~~~~~~~~~~~~~~~~

   As we mentioned in `Chapter
   3 <#Chapter_3._Artificial_Neural_Networks>`__, training a model
   basically means retrieving the model parameters by minimizing the
   cost (loss) function. The two steps for training a linear regression
   model are:

   *Define a cost function (or loss function)*

   Measures how inaccurate the model’s predictions are. The *sum of
   squared residu‐ als (RSS)* as defined in `Equation
   4-1 <#_bookmark196>`__ measures the squared sum of the difference
   between the actual and predicted value and is the cost function for
   linear regression.

   *Equation 4-1. Sum of squared residuals*

   *n n* 2

   *RSS* = ∑ (*y\ i* – *β*\ :sub:`0` – ∑ *β\ j x\ ij*)

   *i*\ =1 *j*\ =1

   In this equation, *β*\ :sub:`0` is the intercept; *β\ j* represents
   the coefficient; *β*\ :sub:`1`, .., *β\ j* are the coefficients of
   the regression; and *x\ ij* represents the *i th* observation and *j
   th* variable.

   *Find the parameters that minimize loss*

   |image22|\ For example, make our model as accurate as possible.
   Graphically, in two dimen‐ sions, this results in a line of best fit
   as shown in `Figure 4-2 <#_bookmark197>`__. In higher dimen‐ sions,
   we would have higher-dimensional hyperplanes. Mathematically, we look
   at the difference between each real data point (*y*) and our model’s
   prediction (*ŷ*). Square these differences to avoid negative numbers
   and penalize larger differ‐ ences, and then add them up and take the
   average. This is a measure of how well our data fits the line.

   *Figure 4-2. Linear regression*

Grid search
~~~~~~~~~~~

   The overall idea of the grid search is to create a grid of all
   possible hyperparameter combinations and train the model using each
   one of them. Hyperparameters are the external characteristic of the
   model, can be considered the model’s settings, and are not estimated
   based on data-like model parameters. These hyperparameters are tuned
   during grid search to achieve better model performance.

   Due to its exhaustive search, a grid search is guaranteed to find the
   optimal parame‐ ter within the grid. The drawback is that the size of
   the grid grows exponentially with the addition of more parameters or
   more considered values.

   The GridSearchCV class in the model_selection module of the sklearn
   package facil‐ itates the systematic evaluation of all combinations
   of the hyperparameter values that we would like to test.

   The first step is to create a model object. We then define a
   dictionary where the key‐ words name the hyperparameters and the
   values list the parameter settings to be tested. For linear
   regression, the hyperparameter is fit_intercept, which is a boolean
   variable that determines whether or not to calculate the *intercept*
   for this model. If set to False, no intercept will be used in
   calculations:

   model = LinearRegression()

   param_grid = {'fit_intercept': [True, False]}

   }

   The second step is to instantiate the GridSearchCV object and provide
   the estimator object and parameter grid, as well as a scoring method
   and cross validation choice, to the initialization method. Cross
   validation is a resampling procedure used to evaluate machine
   learning models, and scoring parameter is the evaluation metrics of
   the model:\ `1 <#_bookmark202>`__

   With all settings in place, we can fit GridSearchCV:

   grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=
   'r2', \\ cv=kfold)

   grid_result = grid.fit(X, Y)

Advantages and disadvantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   In terms of advantages, linear regression is easy to understand and
   interpret. How‐ ever, it may not work well when there is a nonlinear
   relationship between predicted and predictor variables. Linear
   regression is prone to *overfitting* (which we will dis‐ cuss in the
   next section) and when a large number of features are present, it may
   not handle irrelevant features well. Linear regression also requires
   the data to follow cer‐ tain
   `assumptions <https://oreil.ly/tNDnc>`__, such as the absence of
   multicollinearity. If the assumptions fail, then we cannot trust the
   results obtained.
