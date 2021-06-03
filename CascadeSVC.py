from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import numpy as np

class CascadeSVC():
    def __init__(self, fold_size=10000, verbose=True, C=1, kernel="rbf", degree=3,
                 gamma="scale", coef0=0.0, probability=False):
        self.fold_size = fold_size
        self.verbose = verbose
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        base_svc = SVC(C=self.C, kernel=self.kernel, degree=self.degree,
                       gamma=self.gamma, coef0=self.coef0, probability=self.probability)
        self.base_svc = base_svc
        self._estimator_type = "classifier"
    def __get_sv__(self,id,X,y):
        self.base_svc.fit(X,y)
        ind_sv = self.base_svc.support_
        X = X[ind_sv, :]
        y = y[ind_sv]
        id = id[ind_sv]
        return id, X, y
    def __kfold_svc__(self,id,X,y):
        skf = StratifiedKFold(n_splits=round(X.shape[0]/self.fold_size),shuffle=True)
        Xsv = []
        ysv = []
        idsv = []
        for _, ind in skf.split(X,y):
            id_, X_, y_ = self.__get_sv__(id[ind],X[ind,:],y[ind])
            idsv.append(id_)
            Xsv.append(X_)
            ysv.append(y_)
        Xsv = np.vstack(Xsv)
        ysv = np.concatenate(ysv)
        idsv = np.concatenate(idsv)
        return idsv, Xsv, ysv
    def fit(self,X,y):
        self.classes_, y = np.unique(y, return_inverse=True)
        if X.shape[0]<2*self.fold_size:
            print("The number of instances is lower than 2*fold_size")
            print("An only SVC estimation is performed")
            print("The following estimator is used: "+str(self.base_svc))
            id, X, y = self.__get_sv__(np.arange(X.shape[0]), X, y)
            self.n_steps = 1
            self.support_ = id
        else:
            n_init = X.shape[0]
            k = 1
            if self.verbose:
                print("Cascade step "+str(k))
                print("Total number of instances: "+str(n_init))
            id = np.arange(X.shape[0]).astype("int")
            id, X, y = self.__kfold_svc__(id, X, y)
            n_new = X.shape[0]
            if self.verbose:
                print("Number of remaining instances: "+str(n_new))
            while((n_new>2*self.fold_size)&(((n_init-n_new)/n_init)>0.1)):
                k = k+1
                n_init = n_new
                if self.verbose:
                    print("Cascade step " + str(k))
                id, X, y = self.__kfold_svc__(id, X, y)
                n_new = X.shape[0]
                if self.verbose:
                    print("Number of remaining instances: " + str(n_new))
            k = k+1
            if self.verbose:
                print("Cascade step " + str(k))
            id, X, y = self.__get_sv__(id, X, y)
            self.nsteps = k
            self.support_ = id
        if self.verbose:
            print("Final number of support vectors: "+str(len(self.support_)))
    def decision_function(self,X):
        return self.base_svc.decision_function(X)
    def predict(self,X):
        pred = self.classes_[self.base_svc.predict(X)]
        return pred
    def predict_proba(self,X):
        prob = self.base_svc.predict_proba(X)
        return prob
    def get_params(self, deep = True):
        res = {"fold_size" : self.fold_size, "verbose" : self.verbose,
               "C" : self.C, "kernel" : self.kernel, "degree" : self.degree,
               "gamma" : self.gamma, "coef0" : self.coef0,
               "probability" : self.probability}
        return res
    def set_params(self, **parameters):
        for par, val in parameters.items():
            if par in ["fold_size","verbose"]:
                setattr(self, par, val)
            else:
                setattr(self.base_svc, par, val)
        return self
