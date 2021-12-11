.. _ML_Model_Python:

Python for ML and AI in Finance
=========

In terms of the platforms used for machine learning, there are many algorithms and programming languages. However, the Python ecosystem is one of the most domi‐ nant and fastest-growing programming languages for machine learning.
Given the popularity and high adoption rate of Python, we will use it as the main programming language throughout the book. This chapter provides an overview of a Python-based machine learning framework. First, we will review the details of Python-based packages used for machine learning, followed by the model develop‐ ment steps in the Python framework.
The steps of model development in Python presented in this chapter serve as the foundation for the case studies presented in the rest of the book. The Python frame‐ work can also be leveraged while developing any machine learning–based model in finance.
Why Python?
Some reasons for Python’s popularity are as follows:
    • High-level syntax (compared to lower-level languages of C, Java, and C++). Applications can be developed by writing fewer lines of code, making Python attractive to beginners and advanced programmers alike.
    • Efficient development lifecycle.
    • Large collection of community-managed, open-source libraries.
    • Strong portability.
The simplicity of Python has attracted many developers to create new libraries for machine learning, leading to strong adoption of Python.
Python Packages for Machine Learning
The main Python packages used for machine learning are highlighted in Figure 2-1.
Figure 2-1. Python packages
Here is a brief summary of each of these packages:
NumPy
Provides support for large, multidimensional arrays as well as an extensive col‐ lection of mathematical functions.
Pandas
A library for data manipulation and analysis. Among other features, it offers data structures to handle tables and the tools to manipulate them.
Matplotlib
A plotting library that allows the creation of 2D charts and plots.
SciPy
The combination of NumPy, Pandas, and Matplotlib is generally referred to as SciPy. SciPy is an ecosystem of Python libraries for mathematics, science, and engineering.
Scikit-learn (or sklearn)
A machine learning library offering a wide range of algorithms and utilities.
StatsModels
A Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests and statistical data exploration.
TensorFlow and Theano
Dataflow programming libraries that facilitate working with neural networks.
Keras
An artificial neural network library that can act as a simplified interface to TensorFlow/Theano packages.
Seaborn
A data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
pip and Conda
These are Python package managers. pip is a package manager that facilitates installation, upgrade, and uninstallation of Python packages. Conda is a package manager that handles Python packages as well as library dependencies outside of the Python packages.
Python and Package Installation
There are different ways of installing Python. However, it is strongly recommended that you install Python through Anaconda. Anaconda contains Python, SciPy, and Scikit-learn.
.. note:: All code samples in this book use Python 3 and are presented in Jupyter notebooks. Several Python packages, especially Scikit-learn and Keras, are extensively used in the case studies
