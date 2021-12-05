.. _logr:

Regularized Regression
----------------------

   When a linear regression model contains many independent variables,
   their coeffi‐ cients will be poorly determined, and the model will
   have a tendency to fit extremely well to the training data (data used
   to build the model) but fit poorly to testing data (data used to test
   how good the model is). This is known as overfitting or high
   variance.

   One popular technique to control overfitting is *regularization*,
   which involves the addition of a *penalty* term to the error or loss
   function to discourage the coefficients from reaching large values.
   Regularization, in simple terms, is a penalty mechanism that applies
   shrinkage to model parameters (driving them closer to zero) in order
   to build a model with higher prediction accuracy and interpretation.
   Regularized regres‐ sion has two advantages over linear regression:

   *Prediction accuracy*

   The performance of the model working better on the testing data
   suggests that the model is trying to generalize from training data. A
   model with too many parameters might try to fit noise specific to the
   training data. By shrinking or set‐ ting some coefficients to zero,
   we trade off the ability to fit complex models (higher bias) for a
   more generalizable model (lower variance).

   *Interpretation*

   A large number of predictors may complicate the interpretation or
   communica‐ tion of the big picture of the results. It may be
   preferable to sacrifice some detail to limit the model to a smaller
   subset of parameters with the strongest effects.

   The common ways to regularize a linear regression model are as
   follows:

   *L1 regularization or Lasso regression*

   *Lasso regression* performs *L1 regularization* by adding a factor of
   the sum of the absolute value of coefficients in the cost function
   (RSS) for linear regression, as mentioned in `Equation
   4-1 <#_bookmark196>`__. The equation for lasso regularization can be
   repre‐ sented as follows:

   *CostFunction* = *RSS* + *λ* \* ∑ *p* \|\ *β* \|

   L1 regularization can lead to zero coefficients (i.e., some of the
   features are com‐ pletely neglected for the evaluation of output).
   The larger the value of *λ*, the more features are shrunk to zero.
   This can eliminate some features entirely and give us a subset of
   predictors, reducing model complexity. So Lasso regression not only
   helps in reducing overfitting, but also can help in feature
   selection. Predictors not shrunk toward zero signify that they are
   important, and thus L1 regularization allows for feature selection
   (sparse selection). The regularization parameter (*λ*) can be
   controlled, and a lambda value of zero produces the basic linear
   regression equation.

   A lasso regression model can be constructed using the Lasso class of
   the sklearn package of Python, as shown in the code snippet that
   follows:

   from sklearn.linear_model import Lasso model = Lasso()

   model.fit(X, Y)

   *L2 regularization or Ridge regression*

   *Ridge regression* performs *L2 regularization* by adding a factor of
   the sum of the square of coefficients in the cost function (RSS) for
   linear regression, as men‐ tioned in `Equation
   4-1 <#_bookmark196>`__. The equation for ridge regularization can be
   represented as follows:

   *CostFunction* = *RSS* + *λ* \* ∑ *p β* 2

   *j* =1 *j*

   Ridge regression puts constraint on the coefficients. The penalty
   term (*λ*) regu‐ larizes the coefficients such that if the
   coefficients take large values, the optimiza‐ tion function is
   penalized. So ridge regression shrinks the coefficients and helps to
   reduce the model complexity. Shrinking the coefficients leads to a
   lower var‐ iance and a lower error value. Therefore, ridge regression
   decreases the complex‐ ity of a model but does not reduce the number
   of variables; it just shrinks their effect. When *λ* is closer to
   zero, the cost function becomes similar to the linear regression cost
   function. So the lower the constraint (low *λ*) on the features, the
   more the model will resemble the linear regression model.

   A ridge regression model can be constructed using the Ridge class of
   the sklearn package of Python, as shown in the code snippet that
   follows:

   from sklearn.linear_model import Ridge model = Ridge()

   model.fit(X, Y)

   *Elastic net*

   *Elastic nets* add regularization terms to the model, which are a
   combination of both L1 and L2 regularization, as shown in the
   following equation:

   *CostFunction* = *RSS* + *λ* \* ((1 – *α*) / 2 \* ∑ *p β* :sup:`2` +
   *α* \* ∑ *p* \|\ *β* \|)

   In addition to setting and choosing a *λ* value, an elastic net also
   allows us to tune the alpha parameter, where *α* = *0* corresponds to
   ridge and *α* = *1* to lasso. There‐ fore, we can choose an *α* value
   between *0* and *1* to optimize the elastic net. Effec‐ tively, this
   will shrink some coefficients and set some to *0* for sparse
   selection.

   An elastic net regression model can be constructed using the
   ElasticNet class of the sklearn package of Python, as shown in the
   following code snippet:

   from sklearn.linear_model import ElasticNet model = ElasticNet()

   model.fit(X, Y)

   For all the regularized regression, *λ* is the key parameter to tune
   during grid search in Python. In an elastic net, *α* can be an
   additional parameter to tune.