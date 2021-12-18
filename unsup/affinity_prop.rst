.. _affinity_prop:

Affinity Propagation Clustering
-------------------------------

   *Affinity propagation* creates clusters by sending messages between
   data points until convergence. Unlike clustering algorithms such as
   *k*-means, affinity propagation does not require the number of
   clusters to be determined or estimated before running the algorithm.
   Two important parameters are used in affinity propagation to
   determine the number of clusters: the *preference*, which controls
   how many *exemplars* (or proto‐ types) are used; and the *damping
   factor*, which dampens the responsibility and availa‐ bility of
   messages to avoid numerical oscillations when updating these
   messages.

   A dataset is described using a small number of exemplars. These are
   members of the input set that are representative of clusters. The
   affinity propagation algorithm takes in a set of pairwise
   similarities between data points and finds clusters by maximizing the
   total similarity between data points and their exemplars. The
   messages sent between pairs represent the suitability of one sample
   to be the exemplar of the other, which is updated in response to the
   values from other pairs. This updating happens iteratively until
   convergence, at which point the final exemplars are chosen, and we
   obtain the final clustering.

   In terms of strengths, affinity propagation does not require the
   number of clusters to be determined before running the algorithm. The
   algorithm is fast and can be applied to large similarity matrices.
   However, the algorithm often converges to suboptimal solutions, and
   at times it can fail to converge.

.. _implementation-in-python-4:

Implementation in Python
~~~~~~~~~~~~~~~~~~~~~~~~

   The following code snippet illustrates how to implement the affinity
   propagation algorithm for a dataset:

   .. code-block:: python
   
   
      from sklearn.cluster import AffinityPropagation

      *# Initialize the algorithm and set the number of PC's*

      ap = AffinityPropagation() ap.fit(X)

   More details regarding the hyperparameters of affinity propagation
   clustering can be found on the `sklearn
   website <https://scikit-learn.org/>`__. We will look at the affinity
   propagation technique in case studies 1 and 2 in this chapter.