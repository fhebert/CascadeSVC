import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y
from typing import Union
from typing_extensions import Self

class CascadeSVC(ClassifierMixin, BaseEstimator):

    """An implementation of the Cascade SVM algorithm, using the same conventions as the scikit-learn package,
    and can be used as any scikit-learn classification algorithm (same functions fit, predict, can be used in GridSearchCV...)"""

    def __init__(self, fold_size: int = 10000, verbose: bool = True, C: float = 1.0,
                 kernel: str = "rbf", degree: int = 3, gamma: Union[str, float] = "scale",
                 coef0: float = 0.0, probability: bool = False, random_state: Union[int, None] = None) -> None:
        """Initialization:
        - fold_size: the size of folds in which the dataset will be splitted, one SVC estimator being fitted on each fold
        - verbose: if True, prints information during training
        - other parameters: parameters which can be passed to the SVC class"""
        self.fold_size = fold_size
        self.verbose = verbose
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.random_state = random_state

    def _fit_base_svc(self, X: np.array, y: np.array) -> SVC:
        """Fits a base SVC classifier with given data"""
        params = self.get_params()
        del params["fold_size"]
        del params["verbose"]
        base_svc = SVC(**params)
        base_svc.fit(X, y)
        return base_svc

    def _get_support_vectors(self, id: np.array, X: np.array, y: np.array) -> tuple[np.array, np.array, np.array]:
        """Fits a SVC estimator on (X,y) to retrieve the support vectors"""
        base_svc = self._fit_base_svc(X, y)
        ind_sv = base_svc.support_
        id, X, y = id[ind_sv], X[ind_sv, :], y[ind_sv]
        return id, X, y

    def _kfold_svc(self, id: np.array, X: np.array, y: np.array) -> tuple[np.array, np.array, np.array]:
        """Splits data in a number of folds such as each fold contains approximately self.fold_size instances,
        fits a SVC estimator on each fold and retrieves corresponding support vectors; returns arrays containing
        all support vectors"""
        folds = StratifiedKFold(n_splits=round(X.shape[0]/self.fold_size),shuffle=True,random_state=self.random_state)
        idsv, Xsv, ysv = [], [], []
        for _, ind in folds.split(X, y):
            tmp_id, tmp_X, tmp_y = self._get_support_vectors(id[ind],X[ind,:],y[ind])
            idsv.append(tmp_id), Xsv.append(tmp_X), ysv.append(tmp_y)
        idsv, Xsv, ysv = np.hstack(ysv), np.vstack(Xsv), np.hstack(ysv)
        return idsv, Xsv, ysv

    def fit(self, X: np.array, y: np.array) -> Self:
        X, y = check_X_y(X, y)
        self.classes_, y = np.unique(y, return_inverse=True)
        if X.shape[0] < 2*self.fold_size:
            print("The number of instances is lower than 2*fold_size. A single SVC classifier is fitted.")
            self.final_svc_ = self._fit_base_svc(X, y)
            self.nlayers_ = 1
        else:
            n_prev = X.shape[0]
            k = 1
            id = np.arange(X.shape[0])
            if self.verbose:
                print(f"""Cascade layer {k}""")
                print(f"""Total number of instances: {n_prev}""")
            id, X, y = self._kfold_svc(id, X, y)
            n_new = X.shape[0]
            while n_new > 2*self.fold_size and n_new / n_prev < 0.9:
                n_prev = X.shape[0]
                k += 1
                if self.verbose:
                    print(f"""Cascade layer {k}""")
                    print(f"""Total number of instances: {n_new}""")
                id, X, y = self._kfold_svc(id, X, y)
                n_new = X.shape[0]
            k += 1
            if self.verbose:
                print(f"""Cascade layer {k}""")
            self.final_svc_ = self._fit_base_svc(X, y)
            self.nlayers_ = k
        if self.verbose:
            print("Done")
        return self

    def decision_function(self, X: np.array) -> np.array:
        return self.final_svc_.decision_function(X)

    def predict(self, X: np.array) -> np.array:
        return self.classes_[self.final_svc_.predict(X)]

    def predict_proba(self, X: np.array) -> np.array:
        return self.final_svc_.predict_proba(X)