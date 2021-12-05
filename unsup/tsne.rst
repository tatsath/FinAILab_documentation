.. _tsne:

.. automodule:: stable_baselines.common.base_class


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

   from sklearn.manifold import TSNE X_tsne = TSNE().fit_transform(X)