.. _pca:

.. automodule:: stable_baselines.acer


Principal Component Analysis
----------------------------

   The idea of principal component analysis (PCA) is to reduce the
   dimensionality of a dataset with a large number of variables, while
   retaining as much variance in the data as possible. PCA allows us to
   understand whether there is a different representation of the data
   that can explain a majority of the original data points.

   PCA finds a set of new variables that, through a linear combination,
   yield the original variables. The new variables are called *principal
   components* (PCs). These principal components are orthogonal (or
   independent) and can represent the original data. The number of
   components is a hyperparameter of the PCA algorithm that sets the
   target dimensionality.

   The PCA algorithm works by projecting the original data onto the
   principal compo‐ nent space. It then identifies a sequence of
   principal components, each of which aligns with the direction of
   maximum variance in the data (after accounting for var‐ iation
   captured by previously computed components). The sequential
   optimization also ensures that new components are not correlated with
   existing components. Thus the resulting set constitutes an orthogonal
   basis for a vector space.

   |image37|\ The decline in the amount of variance of the original data
   explained by each principal component reflects the extent of
   correlation among the original features. The number of components
   that capture, for example, 95% of the original variation relative to
   the total number of features provides an insight into the linearly
   independent informa‐ tion of the original data. In order to
   understand how PCA works, let’s consider the distribution of data
   shown in `Figure 7-1 <#_bookmark513>`__.

   *Figure 7-1. PCA-1*

   PCA finds a new quadrant system (*y’* and *x’* axes) that is obtained
   from the original through translation and rotation. It will move the
   center of the coordinate system from the original point *(0, 0)* to
   the center of the distribution of data points. It will then move the
   x-axis into the principal axis of variation, which is the one with
   the

   most variation relative to data points (i.e., the direction of
   maximum spread). Then it moves the other axis orthogonally to the
   principal one, into a less important direction of variation.

   |image38|\ `Figure 7-2 <#_bookmark514>`__ shows an example of PCA in
   which two dimensions explain nearly all the variance of the
   underlying data.

   *Figure 7-2. PCA-2*

   These new directions that contain the maximum variance are called
   principal compo‐ nents and are orthogonal to each other by design.

   There are two approaches to finding the principal components: *Eigen
   decomposition*

   and *singular value decomposition* (SVD).

Eigen decomposition
~~~~~~~~~~~~~~~~~~~

   The steps of Eigen decomposition are as follows:

1. First, a covariance matrix is created for the features.

2. Once the covariance matrix is computed, the *eigenvectors* of the
   covariance matrix are calculated.\ `1 <#_bookmark516>`__ These are
   the directions of maximum variance.

3. The *eigenvalues* are then created. They define the magnitude of the
   principal components.

..

   So, for *n* dimensions, there will be an *n* × *n*
   variance-covariance matrix, and as a result, we will have an
   eigenvector of *n* values and *n* eigenvalues.

   Python’s sklearn library offers a powerful implementation of PCA. The
   sklearn.decomposition.PCA function computes the desired number of
   principal components and projects the data into the component space.
   The following code snippet illustrates how to create two principal
   components from a dataset.

   1 `Eigenvectors and eigenvalues <https://oreil.ly/fDaLg>`__ are
   concepts of linear algebra.

   Implementation

   *# Import PCA Algorithm*

   from sklearn.decomposition import PCA

   *# Initialize the algorithm and set the number of PC's*

   pca = PCA(n_components=2) *# Fit the model to data* pca.fit(data)

   *# Get list of PC's*

   pca.components\_

   *# Transform the model to data*

   pca.transform(data)

   *# Get the eigenvalues*

   pca.explained_variance_ratio

   There are additional items, such as *factor loading*, that can be
   obtained using the functions in the sklearn library. Their use will
   be demonstrated in the case studies.

Singular value decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Singular value decomposition (SVD) is factorization of a matrix into
   three matrices and is applicable to a more general case of *m* × *n*
   rectangular matrices.

   If *A* is an *m* × *n* matrix, then SVD can express the matrix as:

   *A* = *UΣV T*

   where *A* is an *m* × *n* matrix, *U* is an *(m* × *m)* orthogonal
   matrix, *Σ* is an *(m* × *n)* nonnegative rectangular diagonal
   matrix, and *V* is an *(n* × *n)* orthogonal matrix. SVD of a given
   matrix tells us exactly how we can decompose the matrix. *Σ* is a
   diagonal matrix with *m* diagonal values called *singular values*.
   Their magnitude indicates how significant they are to preserving the
   information of the original data. *V* contains the principal
   components as column vectors.

   As shown above, both Eigen decomposition and SVD tell us that using
   PCA is effec‐ tively looking at the initial data from a different
   angle. Both will always give the same answer; however, SVD can be
   much more efficient than Eigen decomposition, as it is able to handle
   sparse matrices (those which contain very few nonzero elements). In
   addition, SVD yields better numerical stability, especially when some
   of the features are strongly correlated.

   *Truncated SVD* is a variant of SVD that computes only the largest
   singular values, where the number of computes is a user-specified
   parameter. This method is differ‐ ent from regular SVD in that it
   produces a factorization where the number of col‐ umns is equal to
   the specified truncation. For example, given an *n* × *n* matrix, SVD
   will produce matrices with *n* columns, whereas truncated SVD will
   produce matrices with a specified number of columns that may be less
   than *n*.

   Implementation

   from sklearn.decomposition import TruncatedSVD svd =
   TruncatedSVD(ncomps=20).fit(X)

   In terms of the weaknesses of the PCA technique, although it is very
   effective in reducing the number of dimensions, the resulting
   principal components may be less interpretable than the original
   features. Additionally, the results may be sensitive to the selected
   number of principal components. For example, too few principal compo‐
   nents may miss some information compared to the original list of
   features. Also, PCA may not work well if the data is strongly
   nonlinear.