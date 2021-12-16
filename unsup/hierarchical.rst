.. _hierarchical:

Hierarchical Clustering
-----------------------

   *Hierarchical clustering* involves creating clusters that have a
   predominant ordering from top to bottom. The main advantage of
   hierarchical clustering is that we do not need to specify the number
   of clusters; the model determines that by itself. This

   clustering technique is divided into two types: agglomerative
   hierarchical clustering and divisive hierarchical clustering.

   *Agglomerative hierarchical clustering* is the most common type of
   hierarchical cluster‐ ing and is used to group objects based on their
   similarity. It is a “bottom-up” approach where each observation
   starts in its own cluster, and pairs of clusters are merged as one
   moves up the hierarchy. The agglomerative hierarchical clustering
   algorithm delivers a *local optimum* and proceeds as follows:

1. Make each data point a single-point cluster and form *N* clusters.

2. Take the two closest data points and combine them, leaving *N-1*
   clusters.

3. Take the two closest clusters and combine them, forming *N-2*
   clusters.

4. Repeat step 3 until left with only one cluster.

..

   *Divisive hierarchical clustering* works “top-down” and sequentially
   splits the remain‐ ing clusters to produce the most distinct
   subgroups.

   Both produce *N-1* hierarchical levels and facilitate the clustering
   creation at the level that best partitions data into homogeneous
   groups. We will focus on the more com‐ mon agglomerative clustering
   approach.

   Hierarchical clustering enables the plotting of *dendrograms*, which
   are visualizations of a binary hierarchical clustering. A dendrogram
   is a type of tree diagram showing hierarchical relationships between
   different sets of data. They provide an interesting and informative
   visualization of hierarchical clustering results. A dendrogram con‐
   tains the memory of the hierarchical clustering algorithm, so you can
   tell how the cluster is formed simply by inspecting the chart.

   `Figure 8-1 <#_bookmark585>`__ shows an example of dendrograms based
   on hierarchical clustering. The distance between data points
   represents dissimilarities, and the height of the blocks represents
   the distance between clusters.

   Observations that fuse at the bottom are similar, while those at the
   top are quite dif‐ ferent. With dendrograms, conclusions are made
   based on the location of the vertical axis rather than on the
   horizontal one.

   The advantages of hierarchical clustering are that it is easy to
   implement it, does not require one to specify the number of clusters,
   and it produces dendrograms that are very useful in understanding the
   data. However, the time complexity for hierarchical clustering can
   result in long computation times relative to other algorithms, such
   as *k*-means. If we have a large dataset, it can be difficult to
   determine the correct num‐ ber of clusters by looking at the
   dendrogram. Hierarchical clustering is very sensitive to outliers,
   and in their presence, model performance decreases significantly.

   .. image:: ../_static/img/fig8-1.png
   
   *Figure 8-1. Hierarchical clustering*

.. _implementation-in-python-3:

Implementation in Python
~~~~~~~~~~~~~~~~~~~~~~~~

   The following code snippet illustrates how to apply agglomerative
   hierarchical clus‐ tering with four clusters on a dataset:

   from sklearn.cluster import AgglomerativeClustering

   model = AgglomerativeClustering(n_clusters=4, affinity='euclidean',\\
   linkage='ward')

   clust_labels1 = model.fit_predict(X)

   More details regarding the hyperparameters of agglomerative
   hierarchical clustering can be found on the `sklearn
   website <https://scikit-learn.org/>`__. We will look at the
   hierarchical clustering tech‐ nique in case studies 1 and 3 in this
   chapter.