# Cascade Support Vector Machine Classifier

## Introduction

The CascadeSVC class provides an implementation of the Cascade Support Vector Machine Classifier, as proposed in [Graf, H., Cosatto, E., Bottou, L., Dourdanovic, I., & Vapnik, V. (2004)](https://leon.bottou.org/publications/pdf/nips-2004c.pdf).
This algorithm splits the training dataset in folds and fits an SVM on each fold. Then, only the support vectors obtained on each fold are kept. This is repeated until the dataset made of remaining support vectors is small enough to fit 
a standard SVM on it. This algorithm can thus be more convenient for datasets of very large sample size, for which SVMs are very long to train.
This implementation is based on the [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class available in the [Scikit-Learn](https://scikit-learn.org/stable/index.html) package, and can be used with the 
functionalities of the Scikit-Learn package.

## Importation of the CascadeSVC class

To import the CascadeSVC class, enter the following commands:

```
from urllib.request import urlopen
exec(urlopen("https://raw.githubusercontent.com/fhebert/CascadeSVC/main/CascadeSVC.py").read())
```

## Usage and examples

The CascadeSVC class is very similar to the SVC class available in the Scikit-Learn package. The following arguments
can be used to initialize an instance of the class:
- fold_size: size of the folds (default value: 10000)
- verbose: whether to print messages during the fitting process or not (default value: True)   
- C, kernel, degree, gamma, coef0, probability: arguments of the SVC class, with identical default values.

Like the estimators of the Scikit-Learn package, fitting is performed using the ```fit``` method, and the ```predict```
and ```predict_proba``` methods are used to obtain the predicted labels or probabilities. 

The following function will be used to generate datasets according to several scenarios:
```
def make_dataset(n,scenario):
    if scenario==1:
        n = round(n/2)
        X0 = np.random.normal(np.array([0, 0]), np.array([5, 5]), np.array([n, 2]))
        X1 = np.random.normal(np.array([15, 15]), np.array([5, 5]), np.array([n, 2]))
        X = np.vstack([X0, X1])
        Y = np.array([0] * n + [1] * n)
    if scenario==2:
        X = np.random.normal(np.array([0, 0]), np.array([10, 10]), np.array([n, 2]))
        z = (X[:, 0] ** 2 + X[:, 1] ** 2) - (10 ** 2)
        p = 1 / (1 + np.exp(-5 + 0.1 * z))
        Y = np.random.binomial(n=1, p=p)
        ind = np.where(Y == 0)[0]
        X[ind, :] = X[ind, :] * np.random.normal(np.array([2, 2]),
                                                 np.array([0.2, 0.2]),
                                                 np.array([len(ind), 2]))
        ind = np.where(Y == 1)[0]
        X[ind, :] = np.random.normal(np.array([0, 0]), np.array([10, 10]), np.array([len(ind), 2]))
    if scenario==3:
        n = round(n/3)
        X0 = np.random.normal(np.array([0, 0]), np.array([5, 5]), np.array([n, 2]))
        X1 = np.random.normal(np.array([15, 15]), np.array([5, 5]), np.array([n, 2]))
        X2 = np.random.normal(np.array([-5, 15]), np.array([5, 5]), np.array([n, 2]))
        X = np.vstack([X0, X1, X2])
        Y = np.array([0] * n + [1] * n + [2] * n)
    return X, Y
```

Let us first generate a dataset of size 100,000 according to the first scenario:
```
X, Y = make_dataset(scenario=1,n=100000)
```
The CascadeSVC class will be fitted on this dataset as well as the SVC class.
For each of them, a scaling preprocessing step will be performed using a pipeline.
The computational time of each can be compared.
```
import time
from sklearn.pipeline import Pipeline
csvm = Pipeline([("scaler", StandardScaler()), 
                 ("csvm", CascadeSVC(fold_size=10000,C=0.1,gamma=0.1,kernel="rbf"))])
t0 = time.time()
csvm.fit(X,Y)
t1 = time.time()
print("Elapsed time (seconds): "+str(t1-t0))
svm = Pipeline([("scaler", StandardScaler()), 
                ("svm", SVC(C=0.1,gamma=0.1,kernel="rbf"))])
t0 = time.time()
svm.fit(X,Y)
t1 = time.time()
print("Elapsed time (seconds): "+str(t1-t0))                
```
The CascadeSVC class took 5.5 seconds to fit, while the SVC class took 23.5 seconds. Let us compare the prediction performance:
```
Xtest, Ytest = make_dataset(scenario=1,n=10000)
pred = csvm.predict(Xtest)
np.mean(Ytest==pred)
pred = svm.predict(Xtest)
np.mean(Ytest==pred)
```
In each case, the accuracy equals 0.9837.

The hyperparameters can be tuned using GridSearchCV, using the logistic loss:
```
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import log_loss, make_scorer
log_loss_score = make_scorer(score_func=log_loss,greater_is_better=False,needs_proba=True)

csvm = Pipeline([("scaler",StandardScaler()),
                 ("csvm",CascadeSVC(fold_size=10000,verbose=False,
                                    kernel="rbf",probability=True))])

gridsearch = GridSearchCV(estimator=csvm,
                          param_grid={"csvm__C" : [0.001,0.1,10],
                                      "csvm__gamma" : [0.001,0.1,10]},
                          cv=5,n_jobs=5,verbose=10,scoring=log_loss_score,refit=False)
gridsearch.fit(X_app,Y_app)
par1 = gridsearch.best_params_
res1 = pd.DataFrame({
    "C" : gridsearch.cv_results_["param_csvm__C"],
    "gamma" : gridsearch.cv_results_["param_csvm__gamma"],
    "mean_test_score" : gridsearch.cv_results_["mean_test_score"],
    "std_test_score" : gridsearch.cv_results_["std_test_score"]
})
print(par1)
print(res1)
```
The printed results are as follows:
```
{'csvm__C': 0.1, 'csvm__gamma': 0.1}
     C gamma  mean_test_score  std_test_score
0  0.1   0.1        -0.049435        0.002010
1  0.1     1        -0.273728        0.004995
2  0.1    10        -0.104595        0.002368
3    1   0.1        -0.177840        0.003545
4    1     1        -0.541762        0.015913
5    1    10        -0.120338        0.001726
6   10   0.1        -0.586251        0.016657
7   10     1        -0.301789        0.015065
8   10    10        -0.145069        0.003072
```

Two other scenarios are available in the ```make_dataset``` function: scenario 2 generates a two class dataset with a non-linear 
boundary, while scenario 3 generates a three class dataset with linear boundaries.
