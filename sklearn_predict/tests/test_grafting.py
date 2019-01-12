import pytest
import numpy as np

from sklearn.datasets import load_iris, make_classification
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from sklearn.linear_model import SGDClassifier
from sqlalchemy import create_engine
import pandas as pd

from sklearn_predict.linear_model import GraftingLinearModel

@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_sgd_estimator_iris(data):
    est = SGDClassifier(max_iter = 1000, tol=1e-3, random_state=10)

    X = data[0]
    X2 = data[0][:, :2]
    y = data[1]

    model = GraftingLinearModel(est)
    model.fit(X2, y)
    assert model.score(X2, y)
    with pytest.raises(Exception):
        model.score(X, y)
    
    model.partial_fit(X, y)
    assert model.score(X, y)

    _, cc, _ = model.mask_model(top_k=2)
    assert cc.shape[1] < 4

    _, cc, _ = model.mask_model(alpha=0)
    assert cc.shape[1] == 4

def test_mandelon_estimator_iris(data):
    est = SGDClassifier(max_iter = 1000, tol=1e-3, random_state=10)

    X, y = make_classification()
    X2 = X[:, :10]

    model = GraftingLinearModel(est)
    model.fit(X2, y)
    assert model.score(X2, y)
    assert model.predict(X2).shape[0] == X2.shape[0]
    with pytest.raises(Exception):
        model.predict_proba(X)
    with pytest.raises(Exception):
        model.score(X, y)
    
    model.partial_fit(X, y)
    assert model.score(X, y)


def test_mandelon_estimator_nothing(data):
    est = SGDClassifier(max_iter = 1000, tol=1e-3, random_state=10)

    X, y = make_classification()

    model = GraftingLinearModel(est)
    model.fit(X, y)
    assert model.score(X, y)
    assert model.predict(X).shape[0] == X.shape[0]
    with pytest.raises(Exception):
        model.predict_proba(X)
    
    model.partial_fit(X, y)
    assert model.score(X, y)