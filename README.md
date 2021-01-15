## Introduction

Probability theory and statistical methods play a central role in science. Nowadays we are
surrounded by huge amounts of data. For example, there are about one trillion web pages; more than one
hour of video is uploaded to YouTube every second, amounting to years of content every
day; the genomes of 1000s of people, each of which has a length of more than a billion  base pairs, have
been sequenced by various labs and so on. This deluge of data calls for automated methods of data analysis,
which is exactly what machine learning aims at providing. 

## Learning outcomes

This course aims at giving you insights and knowledge about  many of the central algorithms used in Data Analysis and Machine Learning.  The course is project based and through  various numerical projects, normally three, you will be exposed to fundamental research problems in these fields, with the aim to reproduce state of the art scientific results. Both supervised and unsupervised methods will be covered. The emphasis is on a frequentist approach, although we will try to link it with a Bayesian approach as well. You will learn to develop and structure large codes for studying different cases where Machine Learning is applied to, get acquainted with computing facilities and learn to handle large scientific projects. A good scientific and ethical conduct is emphasized throughout the course. More specifically, after this course you will

- Learn about basic data analysis, statistical analysis, Bayesian statistics, Monte Carlo sampling, data optimization and machine learning;
- Be capable of extending the acquired knowledge to other systems and cases;
- Have an understanding of central algorithms used in data analysis and machine learning;
- Understand linear methods for regression and classification, from ordinary least squares, via Lasso and Ridge to Logistic regression;
- Learn about neural networks and deep  learning methods for supervised and unsupervised learning. Emphasis on feed forward neural networks, convolutional and recurrent neural networks; 
- Learn about about decision trees, random forests, bagging and boosting methods;
- Learn about support vector machines and kernel transformations;
- Reduction of data sets, from PCA to clustering;
- Autoencoders and Reinforcement Learning;
- Work on numerical projects to illustrate the theory. The projects play a central role and you are expected to know modern programming languages like Python or C++ and/or Fortran (Fortran2003 or later).  

## Prerequisites

Basic knowledge in programming and mathematics, with an emphasis on linear algebra. Knowledge of Python or/and C++ as programming languages is strongly recommended and experience with Jupiter notebook is recommended. Required courses are the equivalents to the University of Oslo mathematics courses MAT1100, MAT1110, MAT1120 and at least one of the corresponding computing and programming courses INF1000/INF1110 or MAT-INF1100/MAT-INF1100L/BIOS1100/KJM-INF1100. Most universities offer nowadays a basic programming course (often compulsory) where Python is the recurring programming language.


## The course has two central parts

1. Statistical analysis and optimization of data
2. Machine learning

These topics will be scattered thorughout the course and may not  necessarily be taught separately. Rather, we will often take an approach (during the lectures and project/exercise sessions) where say elements from statistical data analysis are mixed with specific Machine Learning algorithms. 

### Statistical analysis and optimization of data

The following topics will be covered
- Basic concepts, expectation values, variance, covariance, correlation functions and errors;
- Simpler models, binomial distribution, the Poisson distribution, simple and multivariate normal distributions;
- Central elements of Bayesian statistics and modeling;
- Gradient methods for data optimization, 
- Monte Carlo methods, Markov chains, Gibbs sampling and Metropolis-Hastings sampling;
- Estimation of errors and resampling techniques such as the cross-validation, blocking, bootstrapping and jackknife methods;
- Principal Component Analysis (PCA) and its mathematical foundation

### Machine learning

The following topics will be covered:
- Linear Regression and Logistic Regression;
- Neural networks and deep learning, including convolutional and recurrent neural networks
- Decisions trees, Random Forests, Bagging and Boosting
- Support vector machines
- Bayesian linear and logistic regression
- Boltzmann Machines
- Unsupervised learning Dimensionality reduction, from PCA to cluster models

Hands-on demonstrations, exercises and projects aim at deepening your understanding of these topics.

Computational aspects play a central role and you are expected to work
on numerical examples and projects which illustrate the theory and
varous algorithms discussed during the lectures. We recommend strongly
to form small project groups of 2-3 participants, if possible.

## Prerequisites

Basic knowledge in programming and mathematics, with an emphasis on
linear algebra. Knowledge of Python or/and C++ as programming
languages is strongly recommended and experience with Jupiter notebook
is recommended.


## Practicalities

1. Lectures are in the morning, from 9am-1130am.
2. Four hours of laboratory sessions for work on computational projects, from 2pm to 6pm;
3. Lectures and lab sessions will all be at GANIL, starting January 18 at 9am. 
4. Grading scale: Grades are awarded on a scale from A to F, where A is the best grade and F is a fail. We are aiming at having two projects to be handed in. These will graded and should be finalized not later than two weeks after the course is over. Both projects count 50% each of the final grade. We plan to make the grades available not later than March 1, hopefully the grades will be available before that.


## Lecture material
_The link_ https://compphysics.github.io/MLErasmus/doc/web/course.html gives you direct access to the learning material with lectures slides and jupyter notebooks. Videos of the lectures will be added. 


##  Teaching schedule, topics and teachers
### Teachers: Morten Hjorth-Jensen (MHJ), Per-Dimitri B. Sønderland (PDBS), and Kristian Wold (KW)

###  Week 4, January 18-22, 2020
- _Monday Lecture 9am-1130mam_: Introduction to Machine Learning and linear regression (MHJ)
- _Monday Laboratory 2pm-6pm_: Getting familiar with Git, GitHub, installing Python packages and Computational Exercises (PDBS and KW)
- _Tuesday Lecture 10am-2pm_: Linear Regression and Logistic Regression (MHJ)
- _Tuesday Laboratory 10am-2pm:_ Computational Exercises (PDBS and KW), exercise set 2
- _Wednesday Lecture 9am-1130mam_: Regression and Bias-Variance Tradeoff (MHJ)
- _Wednesday Laboratory 2pm-6pm_: Computational Exercises (PDBS and KW), exercise sets 2 and 3
- _Thursday Lecture 9am-1130mam_: Bias-Variance tradeoff, Logistic Regression and Optimization (MHJ)
- _Thursday Laboratory 2pm-6pm_: Computational Exercises (PDBS and KW), exercise sets 2 and 3
- _Friday Lecture 9am-1130mam_:  Logistic Regression and begin Neural Networks (MHJ)
- _Friday Laboratory 2pm-6pm_: Using and installing TensorFlow and Computational Exercises (PDBS and KW), exercise sets 2 and 3 and first project


### Week 5, January 25-29, 2020
- _Monday Lecture 9am-1130mam_:  Neural Networks  (MHJ)
- _Monday Laboratory 2pm-6pm_: Computational Exercises (PDBS and KW) and work on project 1
- _Tuesday Lecture 10am-2pm_:  Neural Networks, back propagation and examples of classification and regression problems (MHJ)
- _Tuesday Laboratory 2pm-6pm:_ Computational Exercises (PDBS and KW) and work on project 1
- _Wednesday Lecture 9am-1130mam_: Decision Trees, Random Forests and Boosting (MHJ)
- _Wednesday Laboratory 2pm-6pm_: Computational Exercises (PDBS and KW), work on project 1
- _Thursday Lecture 9am-1130mam_:  Decision trees, Random Forests and Boosting (MHJ)
- _Thursday Laboratory 2pm-6pm_: Computational Exercises (PDBS and KW), work on project 1
- _Friday Lecture 9am-1130mam_:  Bossting and XGBoost and Summary of course (MHJ), presentation of project 2
- _Friday Laboratory 2pm-6pm_: Computational Exercises (PDBS and KW), work on projects 1 and 2



## Required Technologies

Course participants are expected to have their own laptops/PCs. We use _Git_ as version control software and the usage of providers like _GitHub_, _GitLab_ or similar are strongly recommended.

We will make extensive use of Python as programming language and its
myriad of available libraries.  You will find
Jupyter notebooks invaluable in your work.  You can run _R_
codes in the Jupyter/IPython notebooks, with the immediate benefit of
visualizing your data. You can also use compiled languages like C++,
Rust, Julia, Fortran etc if you prefer. The focus in these lectures will be mainly 
on Python.


If you have Python installed and you feel
pretty familiar with installing different packages, we recommend that
you install the following Python packages via _pip_ as 

* pip install numpy scipy matplotlib ipython scikit-learn mglearn sympy pandas pillow 

For OSX users we recommend, after having installed Xcode, to
install _brew_. Brew allows for a seamless installation of additional
software via for example 

* brew install python3

For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution,
you can use _pip_ as well and simply install Python as 

* sudo apt-get install python3

### Python installers

If you don't want to perform these operations separately and venture
into the hassle of exploring how to set up dependencies and paths, we
recommend two widely used distrubutions which set up all relevant
dependencies for Python, namely 

* Anaconda:https://docs.anaconda.com/, 

which is an open source
distribution of the Python and R programming languages for large-scale
data processing, predictive analytics, and scientific computing, that
aims to simplify package management and deployment. Package versions
are managed by the package management system _conda_. 

* Enthought canopy:https://www.enthought.com/product/canopy/ 

is a Python
distribution for scientific and analytic computing distribution and
analysis environment, available for free and under a commercial
license.

Furthermore, Google's Colab:https://colab.research.google.com/notebooks/welcome.ipynb is a free Jupyter notebook environment that requires 
no setup and runs entirely in the cloud. Try it out!

### Useful Python libraries
Here we list several useful Python libraries we strongly recommend (if you use anaconda many of these are already there)

* _NumPy_:https://www.numpy.org/ is a highly popular library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* _The pandas_:https://pandas.pydata.org/ library provides high-performance, easy-to-use data structures and data analysis tools 
* _Xarray_:http://xarray.pydata.org/en/stable/ is a Python package that makes working with labelled multi-dimensional arrays simple, efficient, and fun!
* _Scipy_:https://www.scipy.org/ (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. 
* _Matplotlib_:https://matplotlib.org/ is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* _Autograd_:https://github.com/HIPS/autograd can automatically differentiate native Python and Numpy code. It can handle a large subset of Python's features, including loops, ifs, recursion and closures, and it can even take derivatives of derivatives of derivatives
* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics. 
* _scikit-learn_:https://scikit-learn.org/stable/ has simple and efficient tools for machine learning, data mining and data analysis
* _TensorFlow_:https://www.tensorflow.org/ is a Python library for fast numerical computing created and released by Google
* _Keras_:https://keras.io/ is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* And many more such as _pytorch_:https://pytorch.org/,  _Theano_:https://pypi.org/project/Theano/ etc 

## Textbooks

_Recommended textbooks_:
- Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, https://www.springer.com/gp/book/9780387310732. This is the main textbook and this course covers chapters 1-7, 11 and 12. 
- Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, The Elements of Statistical Learning, Springer, https://www.springer.com/gp/book/9780387848570. This is a well-known text and serves as additional text.
- Aurelien Geron, Hands‑On Machine Learning with Scikit‑Learn and TensorFlow, O'Reilly, https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/. This text is very useful since it contains many code examples.

The books by Bishop and Hastie et al. can be downloaded for free if you access the university library via an IP number of your home university.



_General learning book on statistical analysis_:
- Christian Robert and George Casella, Monte Carlo Statistical Methods, Springer
- Peter Hoff, A first course in Bayesian statistical models, Springer

_General Machine Learning Books_:
- Kevin Murphy, Machine Learning: A Probabilistic Perspective, MIT Press
- David J.C. MacKay, Information Theory, Inference, and Learning Algorithms, Cambridge University Press
- David Barber, Bayesian Reasoning and Machine Learning, Cambridge University Press 

