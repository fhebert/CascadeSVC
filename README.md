# Python package cascadesvc: an implementation of the Cascade SVM algorithm

## Introduction

This package contains a class named `CascadeSVC`, which provides an implementation of the Cascade SVM algorithm introduced in [Graf, H., Cosatto, E., Bottou, L., Dourdanovic, I., & Vapnik, V. (2004)](https://leon.bottou.org/publications/pdf/nips-2004c.pdf).

It can be installed using ```pip```.

## Algorithm

The Cascade SVM algorithm is designed to train an SVM on large datasets, containing a great number of instances. Indeed, on such datasets, SVMs can be very long to train, as the training requires the computation of dot products (or scalar products) between all pairs of samples, which might represent a very large number of pairs.

This algorithm takes advantage of the fact that the decision function of the SVM algorithm depends only on a (possibly rather small) subset of the input data. Samples contained in this subset are called support vectors.

This algorithm thus consists in splitting the input dataset in smaller subsets. On each subset, a SVM is fitted, and support vectors are retrieved. Concatenating all support vectors yields a new dataset. This new dataset can be splitted again in smaller ones, on which SVMs will be fitted, and so on, until a criterion is reached.

This process can considerably decrease fitting time. Indeed, dot products are only computed within each subset.

## Usage

The CascadeSVC class is very similar to the SVC class available in the Scikit-Learn package, and can be used the same way. The following arguments can be used to initialize an instance of the class:

- ```fold_size```: size of the folds (default value: 10000)
- ```verbose```: whether to print messages during the fitting process or not (default value: True)
- ```C, kernel, degree, gamma, coef0, probability```: arguments of the SVC class, with identical default values.

Like the estimators of the Scikit-Learn package, fitting is performed using the ```fit``` method, and the ```predict``` and ```predict_proba``` methods are used to obtain the predicted labels or probabilities. It can also be tuned using ```GridSearchCV```. Check the ```examples.ipynb``` file to see some usage examples.
