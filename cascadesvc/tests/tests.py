import unittest
import numpy as np
import pickle
from unittest import TestCase
from cascadesvc.cascadesvc import CascadeSVC
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class CascadeSVCTest(TestCase):

    """Implements some tests to check the behavior and results return by the CascadeSVC class"""

    @staticmethod
    def _import_data() -> tuple[np.array, np.array, np.array, np.array]:
        """Imports generated data used to perform tests"""
        data = np.load("cascadesvc/tests/data/data.npy")
        Xtrain, ytrain = data[:,:-1], data[:,-1].flatten()
        data = np.load("cascadesvc/tests/data/testdata.npy")
        Xtest, ytest = data[:,:-1], data[:,-1].flatten()
        return Xtrain, ytrain, Xtest, ytest
    
    @staticmethod
    def _import_id_support_vectors() -> dict:
        """Imports previously determined support vectors corresponding to the 10,000 first instances in
        the training dataset, stored as a dictionary whose keys are named C_x_gamma_y where x and y
        are the values of C and gamma hyperparameters, and each value is a 1d array giving indices of
        corresponding support vectors"""
        with open("cascadesvc/tests/data/support_vectors1.pkl", 'rb') as f:
            res = pickle.load(f)
        return res
 
    def setUp(self):
        super().setUp()
        self.Xtrain, self.ytrain, self.Xtest, self.ytest = self._import_data()

    def tearDown(self):
        super().tearDown()

    def test_1_get_support_vectors(self):
        """Checks that support vectors are the same as expected ones on the 10,000 first instances of the
        dataset, saved in support_vectors1.pkl"""
        print("Test 1: check that observed and expected support vectors are the same")
        expected_sv = self._import_id_support_vectors()
        X, y = self.Xtrain[:10000,:], self.ytrain[:10000]
        for key in expected_sv.keys():
            C = float(key.split("C_")[1].split("_")[0])
            gamma = float(key.split("gamma_")[1])
            tmp = CascadeSVC(C=C,gamma=gamma)
            id, _, _ = tmp._get_support_vectors(np.arange(X.shape[0]), X, y)
            assert np.all(expected_sv[key] == id), (
                f"Observed support vectors differ from expected support vectors for parameters C = {C} and gamma = {gamma}"
            )
        print("Test passed.")

    def test_2_check_auc(self):
        """Checks the performance on test data with two sets of hyperparameters"""
        print("Test 2: check AUC on test data")
        svm1 = Pipeline(
            (
                ("scaler", StandardScaler()),
                ("svm", CascadeSVC(C = 10, gamma = 0.1, probability = True, verbose = False))
            )
        )
        svm1.fit(self.Xtrain, self.ytrain)
        svm2 = Pipeline(
            (
                ("scaler", StandardScaler()),
                ("svm", CascadeSVC(C = 0.1, gamma = 0.1, probability = True, verbose = False))
            )
        )
        svm2.fit(self.Xtrain, self.ytrain)
        p1 = svm1.predict_proba(self.Xtest)[:,1]
        auc1 = round(roc_auc_score(self.ytest, p1),2)
        p2 = svm2.predict_proba(self.Xtest)[:,1]
        auc2 = round(roc_auc_score(self.ytest, p2),2)
        assert auc1 >= 0.99, f"AUC did not reach the expected value (at least 0.99); reached {auc1}"
        assert auc2 == 0.67, f"AUC did not reach the expected value (0.67); reached {auc2}"
        print("Test passed.")

if __name__ == "__main__":
    unittest.main()
    

    