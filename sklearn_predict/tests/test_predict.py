import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from sklearn.linear_model import SGDClassifier
from sqlalchemy import create_engine
import pandas as pd

from sklearn_predict import linearsvm_sql

@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_sgd_estimator(data):
    est = SGDClassifier(max_iter = 1000, tol=1e-3)
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

    queries = linearsvm_sql(est, iris['feature_names'])
    assert len(queries) == 3

    select_query = ','.join(["{} as p{}".format(x, idx) for idx, x in enumerate(queries)])
    sql_query = "SELECT {} from iris".format(select_query)
    data_out = pd.read_sql(sql_query, engine)
    print(data_out.columns)
    print(y_pred)
    print(data_out)
    data_out['pred0'] = (y_pred == 0)
    data_out['pred1'] = (y_pred == 1)
    data_out['pred2'] = (y_pred == 2)
    assert_array_equal(data_out['pred0'], data_out['p0']>0)
    assert_array_equal(data_out['pred1'], data_out['p1']>0)
    assert_array_equal(data_out['pred2'], data_out['p2']>0)

