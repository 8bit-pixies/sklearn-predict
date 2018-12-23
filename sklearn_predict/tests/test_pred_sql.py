import pytest
import numpy as np

from sklearn.datasets import load_iris, make_classification
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from sklearn.linear_model import SGDClassifier
from sqlalchemy import create_engine
import pandas as pd

from sklearn_predict.linear_model import LinearSVMSQL

@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_iris_query(data):
    est = SGDClassifier(max_iter = 1000, tol=1e-3, random_state=10)
    est.fit(*data)
    assert hasattr(est, 'coef_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred[:10], data[1][:10])

    # build iris table in sqlite...
    engine = create_engine('sqlite://', echo=False)
    iris = load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                columns= iris['feature_names'] + ['target'])
    data.to_sql('iris', con=engine)

    sql_query = LinearSVMSQL(est, iris['feature_names']).to_sql("iris")
    data_out = pd.read_sql(sql_query, engine)
    print(data_out)
    assert "prediction" in data_out.columns
    assert "pred_0" in data_out.columns
    assert "pred_1" in data_out.columns
    assert "pred_2" in data_out.columns


def test_mandelon_query():
    X, y = make_classification()
    est = SGDClassifier(max_iter = 1000, tol=1e-3)
    est.fit(X, y)
    assert hasattr(est, 'coef_')

    y_pred = est.predict(X)
    #assert_array_equal(y_pred[:10], y[:10])

    # build iris table in sqlite...
    engine = create_engine('sqlite://', echo=False)
    feature_names = ["c{}".format(x) for x in range(20)] + ['target']
    data = pd.DataFrame(data= np.c_[X, y],
                columns= feature_names)
    data.to_sql('iris', con=engine)

    sql_query = LinearSVMSQL(est, feature_names).to_sql(tbl_name="iris", pred_prefix="", pred_target=["alert"])
    data_out = pd.read_sql(sql_query, engine)
    print(data_out)
    assert "alert" in data_out.columns
