.. _KernelPCA:

Kernel Principal Component Analysis
-----------------------------------

   A main limitation of PCA is that it only applies linear
   transformations. Kernel principal component analysis (KPCA) extends
   PCA to handle nonlinearity. It first maps the original data to some
   nonlinear feature space (usually one of higher dimension). Then it
   applies PCA to extract the principal components in that space.

   Linear transformations are suitable
   for the blue and red data points on the left-hand plot. However, if
   all dots are arranged as per the graph on the right, the result is
   not linearly separable. We would then need to apply KPCA to
   separate the components.

   .. image:: ../_static/img/fig7-3.png
   

   *Figure 7-3. Kernel PCA*

   Implementation

   from sklearn.decomposition import KernelPCA

   kpca = KernelPCA(n_components=4, kernel='rbf').fit_transform(X)

   In the Python code, we specify kernel='rbf', which is the `radial
   basis function kernel. This is commonly used as a kernel
   in machine learning techniques, such as in SVMs.

   Using KPCA, component separation becomes easier in a higher
   dimensional space, as mapping into a higher dimensional space often
   provides greater classification power.

t-distributed Stochastic Neighbor Embedding
-------------------------------------------

   t-distributed stochastic neighbor embedding (t-SNE) is a
   dimensionality reduction algorithm that reduces the dimensions by
   modeling the probability distribution of neighbors around each point.
   Here, the term *neighbors* refers to the set of points clos‐ est to a
   given point. The algorithm emphasizes keeping similar points together
   in low dimensions as opposed to maintaining the distance between
   points that are apart in high dimensions.

   The algorithm starts by calculating the probability of similarity of
   data points in cor‐ responding high and low dimensional space. The
   similarity of points is calculated as the conditional probability
   that a point *A* would choose point *B* as its neighbor if neighbors
   were picked in proportion to their probability density under a normal
   dis‐ tribution centered at *A*. The algorithm then tries to minimize
   the difference between these conditional probabilities (or
   similarities) in the high and low dimensional spaces for a perfect
   representation of data points in the low dimensional space.

   Implementation

   .. code-block:: python
   
      from sklearn.manifold import TSNE X_tsne = TSNE().fit_transform(X)