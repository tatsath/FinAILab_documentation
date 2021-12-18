.. _svm:

Support Vector Machine
----------------------

   The objective of the *support vector machine* (SVM) algorithm is to
   maximize the mar‐ gin (shown as shaded area in `Figure
   4-3 <#_bookmark220>`__), which is defined as the distance between the
   separating hyperplane (or decision boundary) and the training samples
   that are closest to this hyperplane, the so-called support vectors.
   The margin is calculated as the perpendicular distance from the line
   to only the closest points, as shown in `Figure
   4-3 <#_bookmark220>`__. Hence, SVM calculates a maximum-margin
   boundary that leads to a homogeneous partition of all data points.

   .. image:: ../_static/img/fig4-3.jpg
   

   *Figure 4-3. Support vector machine*

   In practice, the data is messy and cannot be separated perfectly with
   a hyperplane. The constraint of maximizing the margin of the line
   that separates the classes must be relaxed. This change allows some
   points in the training data to violate the separating line. An
   additional set of coefficients is introduced that give the margin
   wiggle room in each dimension. A tuning parameter is introduced,
   simply called *C*, that defines the magnitude of the wiggle allowed
   across all dimensions. The larger the value of *C*, the more
   violations of the hyperplane are permitted.

   In some cases, it is not possible to find a hyperplane or a linear
   decision boundary, and kernels are used. A kernel is just a
   transformation of the input data that allows the SVM algorithm to
   treat/process the data more easily. Using kernels, the original data
   is projected into a higher dimension to classify the data better.

   SVM is used for both classification and regression. We achieve this
   by converting the original optimization problem into a dual problem.
   For regression, the trick is to reverse the objective. Instead of
   trying to fit the largest possible street between two classes while
   limiting margin violations, SVM regression tries to fit as many
   instances as possible on the street (shaded area in `Figure
   4-3 <#_bookmark220>`__) while limiting margin violations. The width
   of the street is controlled by a hyperparameter.

   The SVM regression and classification models can be constructed using
   the sklearn package of Python, as shown in the following code
   snippets:

   Regression

   .. code-block:: python
   
      from sklearn.svm import SVR model = SVR()
      model.fit(X, Y)

   Classification

   .. code-block:: python
   
      from sklearn.svm import SVC model = SVC()
      model.fit(X, Y)

.. _hyperparameters-2:

Hyperparameters
~~~~~~~~~~~~~~~

   The following key parameters are present in the sklearn
   implementation of SVM and can be tweaked while performing the grid
   search:

   *Kernels (*\ kernel *in sklearn)*

   The choice of kernel controls the manner in which the input variables
   will be projected. There are many kernels to choose from, but
   *linear* and `RBF <https://oreil.ly/XpBOi>`__ are the most common.

   *Penalty (*\ C *in sklearn)*

   The penalty parameter tells the SVM optimization how much you want to
   avoid misclassifying each training example. For large values of the
   penalty parameter, the optimization will choose a smaller-margin
   hyperplane. Good values might be a log scale from 10 to 1,000.

.. _advantages-and-disadvantages-2:

Advantages and disadvantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   In terms of advantages, SVM is fairly robust against overfitting,
   especially in higher dimensional space. It handles the nonlinear
   relationships quite well, with many ker‐ nels to choose from. Also,
   there is no distributional requirement for the data.

   In terms of disadvantages, SVM can be inefficient to train and
   memory-intensive to run and tune. It doesn’t perform well with large
   datasets. It requires the feature scal‐ ing of the data. There are
   also many hyperparameters, and their meanings are often not
   intuitive.