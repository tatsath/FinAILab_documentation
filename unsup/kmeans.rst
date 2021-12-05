.. _kmeans:

k-means Clustering
------------------

   *k*-means is the most well-known clustering technique. The algorithm
   of *k*-means aims to find and group data points into classes that
   have high similarity between them. This similarity is understood as
   the opposite of the distance between data points. The closer the data
   points are, the more likely they are to belong to the same cluster.

   The algorithm finds *k* centroids and assigns each data point to
   exactly one cluster with the goal of minimizing the within-cluster
   variance (called *inertia*). It typically uses the Euclidean distance
   (ordinary distance between two points), but other dis‐ tance metrics
   can be used. The *k*-means algorithm delivers a local optimum for a
   given *k* and proceeds as follows:

1. This algorithm specifies the number of clusters.

2. Data points are randomly selected as cluster centers.

3. Each data point is assigned to the cluster center it is nearest to.

4. Cluster centers are updated to the mean of the assigned points.

5. Steps 3–4 are repeated until all cluster centers remain unchanged.

..

   In simple terms, we randomly move around the specified number of
   centroids in each iteration, assigning each data point to the closest
   centroid. Once we have done that, we calculate the mean distance of
   all points in each centroid. Then, once we can no longer reduce the
   minimum distance from data points to their respective cent‐ roids, we
   have found our clusters.

k-means hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~

   The *k*-means hyperparameters include:

   *Number of clusters*

   The number of clusters and centroids to generate.

   *Maximum iterations*

   Maximum iterations of the algorithm for a single run.

   *Number initial*

   The number of times the algorithm will be run with different centroid
   seeds. The final result will be the best output of the defined number
   of consecutive runs, in terms of inertia.

   With *k*-means, different random starting points for the cluster
   centers often result in very different clustering solutions.
   Therefore, the *k*-means algorithm is run in sklearn with at least 10
   different random initializations, and the solution occurring the
   great‐ est number of times is chosen.

   The strengths of *k*-means include its simplicity, wide range of
   applicability, fast con‐ vergence, and linear scalability to large
   data while producing clusters of an even size. It is most useful when
   we know the exact number of clusters, *k*, beforehand. In fact, a
   main weakness of *k*-means is having to tune this hyperparameter.
   Additional draw‐ backs include the lack of a guarantee to find a
   global optimum and its sensitivity to outliers.

.. _implementation-in-python-2:

Implementation in Python
~~~~~~~~~~~~~~~~~~~~~~~~

   Python’s sklearn library offers a powerful implementation of
   *k*-means. The following code snippet illustrates how to apply
   *k*-means clustering on a dataset:

   from sklearn.cluster import KMeans

   *#Fit with k-means*

   k_means = KMeans(n_clusters=nclust) k_means.fit(X)

   The number of clusters is the key hyperparameter to be tuned. We will
   look at the *k*- means clustering technique in case studies 1 and 2
   of this chapter, in which further details on choosing the right
   number of clusters and detailed visualizations are provided.