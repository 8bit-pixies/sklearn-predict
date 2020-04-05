import pytest
import numpy as np

import pandas as pd
from sklearn.utils.testing import assert_array_equal

from sklearn_predict.feature_engineer import OneHotEncoderSQL, SkopeRulesSQL
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder

from skrules import SkopeRules
from sklearn.datasets import make_classification


def test_ohe():
    enc = OneHotEncoderSQL(handle_unknown="ignore")
    X = [["Male", 1], ["Female", 3], ["Female", 2]]
    enc.fit(X)

    export = OneHotEncoderSQL(enc, ["gender", "class"])
    # print(export.export())

    engine = create_engine("sqlite://", echo=False)
    df = pd.DataFrame(X, columns=["gender", "class"])
    df.to_sql("tbl", engine)

    select_query = ",".join(export.export())
    df_out = pd.read_sql("SELECT {} from tbl".format(select_query), engine)

    assert_array_equal(np.array(df_out), enc.transform(X).todense())


def test_skopt():
    X, y = make_classification()
    feature_names = ["c{}".format(idx) for idx in range(20)]
    clf = SkopeRules(
        max_depth_duplication=2,
        n_estimators=30,
        precision_min=0.3,
        recall_min=0.1,
        feature_names=feature_names,
    )
    clf.fit(X, y)
    print(clf.rules_)
    export = SkopeRulesSQL(clf, feature_names)
    print(export.export())

    engine = create_engine("sqlite://", echo=False)
    df = pd.DataFrame(X, columns=feature_names)
    df.to_sql("tbl", engine)

    query_sql = [
        "{} end as rule{}".format(x, idx) for idx, x in enumerate(export.export())
    ]
    df_out = pd.read_sql("select {} from tbl".format(",".join(query_sql)), engine)
    assert df_out.shape[0] == df.shape[0]
