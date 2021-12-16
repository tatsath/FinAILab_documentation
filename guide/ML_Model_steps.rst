.. _ML_Model_steps:

Modeling steps
=========
Steps for Model Development in Python Ecosystem
Working through machine learning problems from end to end is critically important. Applied machine learning will not come alive unless the steps from beginning to end are well defined.
Figure 2-2 provides an outline of the simple seven-step machine learning project template that can be used to jump-start any machine learning model in Python. The
first few steps include exploratory data analysis and data preparation, which are typi‐ cal data science–based steps aimed at extracting meaning and insights from data. These steps are followed by model evaluation, fine-tuning, and finalizing the model.
Figure 2-2. Model development steps

Model Development Blueprint
The following section covers the details of each model development step with sup‐ porting Python code.
    1. Problem definition
The first step in any project is defining the problem. Powerful algorithms can be used for solving the problem, but the results will be meaningless if the wrong problem is solved.
The following framework should be used for defining the problem:
        1. Describe the problem informally and formally. List assumptions and similar problems.
        2. List the motivation for solving the problem, the benefits a solution provides, and how the solution will be used.
        3. Describe how the problem would be solved using the domain knowledge.
    2. Loading the data and packages
The second step gives you everything needed to start working on the problem. This includes loading libraries, packages, and individual functions needed for the model  development.
        2.1. Load libraries. A sample code for loading libraries is as follows:
# Load libraries
import pandas as pd
from matplotlib import pyplot
The details of the libraries and modules for specific functionalities are defined further in the individual case studies.
        2.2. Load data. The following items should be checked and removed before loading the data:
    • Column headers
    • Comments or special characters
    • Delimiter

There are many ways of loading data. Some of the most common ways are as follows:
Load CSV files with Pandas
from pandas import read_csv filename = 'xyz.csv'
data = read_csv(filename, names=names)
Load file from URL
from pandas import read_csv url = 'https://goo.gl/vhm1eU' names = ['age', 'class']
data = read_csv(url, names=names)
Load file using pandas_datareader
import pandas_datareader.data as web

ccy_tickers = ['DEXJPUS', 'DEXUSUK'] idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

stk_data = web.DataReader(stk_tickers, 'yahoo') ccy_data = web.DataReader(ccy_tickers, 'fred') idx_data = web.DataReader(idx_tickers, 'fred')
    3. Exploratory data analysis
In this step, we look at the dataset.
        3.1. Descriptive statistics.	Understanding the dataset is one of the most  important steps of model development. The steps to understanding data include:
            1. Viewing the raw data.
            2. Reviewing the dimensions of the dataset.
            3. Reviewing the data types of attributes.
            4. Summarizing the distribution, descriptive statistics, and relationship among the variables in the dataset.

These steps are demonstrated below using sample Python code:
Viewing the data
set_option('display.width', 100) dataset.head(1)
Output

 	Age  Sex	Job Housing SavingAccounts CheckingAccount CreditAmount Duration Purpose Risk 
0   67	male  2	own	NaN	little	1169	6	radio/TV good

Reviewing the dimensions of the dataset
dataset.shape
Output
(284807, 31)
The results show the dimension of the dataset and mean that the dataset has 284,807 rows and 31 columns.
Reviewing the data types of the attributes in the data
# types set_option('display.max_rows', 500) dataset.dtypes
Summarizing the data using descriptive statistics
# describe data set_option('precision', 3) dataset.describe()
Output


Age
Job
CreditAmount
Duration
count
1000.000
1000.000
1000.000
1000.000
mean
35.546
1.904
3271.258
20.903
std
11.375
0.654
2822.737
12.059
min
19.000
0.000
250.000
4.000
25%
27.000
2.000
1365.500
12.000
50%
33.000
2.000
2319.500
18.000
75%
42.000
2.000
3972.250
24.000
max
75.000
3.000
18424.000
72.000

3.2. Data visualization. The fastest way to learn more about the data is to visualize it. Visualization involves independently understanding each attribute of the dataset.
Some of the plot types are as follows:
Univariate plots
Histograms and density plots
Multivariate plots
Correlation matrix plot and scatterplot
The Python code for univariate plot types is illustrated with examples below:
Univariate plot: histogram
from matplotlib import pyplot
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1,\ figsize=(10,4))
pyplot.show()
Univariate plot: density plot
from matplotlib import pyplot
dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False,\ legend=True, fontsize=1, figsize=(10,4))
pyplot.show()

.. image:: ../_static/img/fig2-3.png


Figure 2-3 illustrates the output.
Figure 2-3. Histogram (top) and density plot (bottom)
The Python code for multivariate plot types is illustrated with examples below:
Multivariate plot: correlation matrix plot
from matplotlib import pyplot import seaborn as sns correlation = dataset.corr() pyplot.figure(figsize=(5,5))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
Multivariate plot: scatterplot matrix
from pandas.plotting import scatter_matrix scatter_matrix(dataset)
Figure 2-4 illustrates the output.

.. image:: ../_static/img/fig2.4.jpg


Figure 2-4. Correlation (left) and scatterplot (right)
    4. Data preparation
Data preparation is a preprocessing step in which data from one or more sources is cleaned and transformed to improve its quality prior to its use.
        4.1. Data cleaning. In machine learning modeling, incorrect data can be costly. Data cleaning involves checking the following:
Validity
The data type, range, etc.
Accuracy
The degree to which the data is close to the true values.
Completeness
The degree to which all required data is known.
Uniformity
The degree to which the data is specified using the same unit of measure.
The different options for performing data cleaning include:
Dropping “NA” values within data
dataset.dropna(axis=0)
Filling “NA” with 0
dataset.fillna(0)
Filling NAs with the mean of the column
dataset['col'] = dataset['col'].fillna(dataset['col'].mean())
        4.2. Feature selection. The data features used to train the machine learning models have a huge influence on the performance. Irrelevant or partially relevant features  can negatively impact model performance. Feature selection1 is a process in which features in data that contribute most to the prediction variable or output are auto‐ matically selected.
The benefits of performing feature selection before modeling the data are:
Reduces overfitting2
Less redundant data means fewer opportunities for the model to make decisions based on noise.
Improves performance
Less misleading data means improved modeling performance.
Reduces training time and memory footprint
Less data means faster training and lower memory footprint.
The following sample feature is an example demonstrating when the best two fea‐ tures are selected using the SelectKBest function under sklearn. The SelectKBest function scores the features using an underlying function and then removes all but  the k highest scoring feature:
from sklearn.feature_selection import SelectKBest from sklearn.feature_selection import chi2 bestfeatures = SelectKBest( k=5)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_) dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1) print(featureScores.nlargest(2,'Score')) #print 2 best features

Output


Specs
Score
2
Variable1
58262.490
3
Variable2
321.031
When features are irrelevant, they should be dropped. Dropping the irrelevant fea‐ tures is illustrated in the following sample code:
#dropping the old features
dataset.drop(['Feature1','Feature2','Feature3'],axis=1,inplace=True)



    
4.3. Data transformation. Many machine learning algorithms make assumptions about the data. It is a good practice to perform the data preparation in such a way that exposes the data in the best possible manner to the machine learning algorithms. This can be accomplished through data transformation.
The different data transformation approaches are as follows:
Rescaling
When data comprises attributes with varying scales, many machine learning algorithms can benefit from rescaling all the attributes to the same scale. Attributes are often rescaled in the range between zero and one. This is useful for optimization algorithms used in the core of machine learning algorithms, and it also helps to speed up the calculations in an algorithm:
from sklearn.preprocessing import MinMaxScaler scaler = MinMaxScaler(feature_range=(0, 1)) rescaledX = pd.DataFrame(scaler.fit_transform(X))
Standardization
Standardization is a useful technique to transform attributes to a standard nor‐ mal distribution with a mean of zero and a standard deviation of one. It is most suitable for techniques that assume the input variables represent a normal distri‐ bution:
from sklearn.preprocessing import StandardScaler scaler = StandardScaler().fit(X)
StandardisedX = pd.DataFrame(scaler.fit_transform(X))
Normalization
Normalization refers to rescaling each observation (row) to have a length of one (called a unit norm or a vector). This preprocessing method can be useful for sparse datasets of attributes of varying scales when using algorithms that weight input values:
from sklearn.preprocessing import Normalizer scaler = Normalizer().fit(X)
NormalizedX = pd.DataFrame(scaler.fit_transform(X))

    5. Evaluate models
Once we estimate the performance of our algorithm, we can retrain the final algo‐ rithm on the entire training dataset and get it ready for operational use. The best way to do this is to evaluate the performance of the algorithm on a new dataset. Different machine learning techniques require different evaluation metrics. Other than model performance, several other factors such as simplicity, interpretability, and training time are considered when selecting a model. The details regarding these factors are covered in Chapter 4.
        5.1. Training and test split. The simplest method we can use to evaluate the perfor‐ mance of a machine learning algorithm is to use different training and testing data‐ sets. We can take our original dataset and split it into two parts: train the algorithm on the first part, make predictions on the second part, and evaluate the predictions against the expected results. The size of the split can depend on the size and specifics of the dataset, although it is common to use 80% of the data for training and the remaining 20% for testing. The differences in the training and test datasets can result in meaningful differences in the estimate of accuracy. The data can easily be split into the training and test sets using the train_test_split function available in sklearn:
# split out validation dataset for the end
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation =\
train_test_split(X, Y, test_size=validation_size, random_state=seed)

        5.2. Identify evaluation metrics. Choosing which metric to use to evaluate machine learning algorithms is very important. An important aspect of evaluation metrics is the capability to discriminate among model results. Different types of evaluation metrics used for different kinds of ML models are covered in detail across several chapters of this book.
        5.3. Compare models and algorithms. Selecting a machine learning model or algorithm is both an art and a science. There is no one solution or approach that fits all. There are several factors over and above the model performance that can impact the deci‐ sion to choose a machine learning algorithm.
Let’s understand the process of model comparison with a simple example. We define two variables, X and Y, and try to build a model to predict Y using X. As a first step, the data is divided into training and test split as mentioned in the preceding section:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split validation_size = 0.2
seed = 7
X = 2 - 3 * np.random.normal(0, 1, 20)
Y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.exp(-X)+np.random.normal(-3, 3, 20)
# transforming the data to include another axis
X = X[:, np.newaxis] Y = Y[:, np.newaxis]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\ test_size=validation_size, random_state=seed)
We have no idea which algorithms will do well on this problem. Let’s design our test now. We will use two models—one linear regression and the second polynomial regression to fit Y against X. We will evaluate algorithms using the Root Mean
Squared Error (RMSE) metric, which is one of the measures of the model perfor‐ mance. RMSE will give a gross idea of how wrong all predictions are (zero is perfect):
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

model = LinearRegression() model.fit(X_train, Y_train) Y_pred = model.predict(X_train)

rmse_lin = np.sqrt(mean_squared_error(Y_train,Y_pred)) r2_lin = r2_score(Y_train,Y_pred)
print("RMSE for Linear Regression:", rmse_lin)

polynomial_features= PolynomialFeatures(degree=2) x_poly = polynomial_features.fit_transform(X_train)

model = LinearRegression() model.fit(x_poly, Y_train) Y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y_train,Y_poly_pred)) r2 = r2_score(Y_train,Y_poly_pred)
print("RMSE for Polynomial Regression:", rmse)
Output
RMSE for Linear Regression: 6.772942423315028 RMSE for Polynomial Regression: 6.420495127266883
We can see that the RMSE of the polynomial regression is slightly better than that of the linear regression.3 With the former having the better fit, it is the preferred model in this step.
    6. Model tuning
Finding the best combination of hyperparameters of a model can be treated as a  search problem.4 This searching exercise is often known as model tuning and is one of the most important steps of model development. It is achieved by searching for the best parameters of the model by using techniques such as a grid search. In a grid search, you create a grid of  all  possible  hyperparameter  combinations  and  train the model using each one of them. Besides a grid search, there are several other


    3 It should be noted that the difference in RMSE is small in this case and may not replicate with a different split of the train/test data.
    4 Hyperparameters are the external characteristics of the model, can be considered the model’s settings, and are not estimated based on data-like model parameters.
techniques for model tuning, including randomized search, Bayesian optimization, and hyperbrand.
In the case studies presented in this book, we focus primarily on grid search for  model tuning.
Continuing on from the preceding example, with the polynomial as the best model: next, run a grid search for the model, refitting the polynomial regression with differ‐ ent degrees. We compare the RMSE results for all the models:
Deg= [1,2,3,6,10]
results=[] names=[]
for deg in Deg:
polynomial_features= PolynomialFeatures(degree=deg) x_poly = polynomial_features.fit_transform(X_train)

model = LinearRegression() model.fit(x_poly, Y_train) Y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y_train,Y_poly_pred)) r2 = r2_score(Y_train,Y_poly_pred)
results.append(rmse) names.append(deg)
plt.plot(names, results,'o') plt.suptitle('Algorithm Comparison')
Output


The RMSE decreases when the degree increases, and the lowest RMSE is for the model with degree 10. However, models with degrees lower than 10 performed very well, and the test set will be used to finalize the best model.
While the generic set of input parameters for each algorithm provides a starting point for analysis, it may not have the optimal configurations for the particular dataset and business problem.
    7. Finalize the model
Here, we perform the final steps for selecting the model. First, we run predictions on the test dataset with the trained model. Then we try to understand the model intu‐  ition and save it for further usage.
        7.1. Performance on the test set. The model selected during the training steps is further evaluated on the test set. The test set allows us to compare different models in an unbiased way, by basing the comparisons in data that were not used in any part of the training. The test results for the model developed in the previous step are shown in the following example:
Deg= [1,2,3,6,8,10]
for deg in Deg:
polynomial_features= PolynomialFeatures(degree=deg) x_poly = polynomial_features.fit_transform(X_train) model = LinearRegression()
model.fit(x_poly, Y_train)
x_poly_test = polynomial_features.fit_transform(X_test) Y_poly_pred_test = model.predict(x_poly_test)
rmse = np.sqrt(mean_squared_error(Y_test,Y_poly_pred_test)) r2 = r2_score(Y_test,Y_poly_pred_test) results_test.append(rmse)
names_test.append(deg) plt.plot(names_test, results_test,'o') plt.suptitle('Algorithm Comparison')
Output

In the training set we saw that the RMSE decreases with an increase in the degree of polynomial model, and the polynomial of degree 10 had the lowest RMSE. However, as shown in the preceding output for the polynomial of degree 10, although the train‐ ing set had the best results, the results in the test set are poor. For the polynomial of degree 8, the RMSE in the test set is relatively higher. The polynomial of degree 6 shows the best result in the test set (although the difference is small compared to other lower-degree polynomials in the test set) as well as good results in the training set. For these reasons, this is the preferred model.
In addition to the model performance, there are several other factors to consider when selecting a model, such as simplicity, interpretability, and training time. These factors will be covered in the upcoming chapters.
        7.2. Model/variable intuition. This step involves considering a holistic view of the approach taken to solve the problem, including the model’s limitations as it relates to the desired outcome, the variables used, and the selected model parameters. Details  on model and variable intuition regarding different types of machine learning models are presented in the subsequent chapters and case studies.
        7.3. Save/deploy. After finding an accurate machine learning model, it must be saved and loaded in order to ensure its usage later.
Pickle is one of the packages for saving and loading a trained model in Python. Using pickle operations, trained machine learning models can be saved in the serialized for‐ mat to a file. Later, this serialized file can be loaded to de-serialize the model for its usage. The following sample code demonstrates how to save the model to a file and load it to make predictions on new data:
# Save Model Using Pickle from pickle import dump from pickle import load
# save the model to disk
filename = 'finalized_model.sav' dump(model, open(filename, 'wb')) # load the model from disk loaded_model = load(filename)

In recent years, frameworks such as AutoML have been built to automate the maximum number of steps in a machine learning model development process. Such frameworks allow the model developers to build ML models with high scale, efficiency, and pro‐ ductivity. Readers are