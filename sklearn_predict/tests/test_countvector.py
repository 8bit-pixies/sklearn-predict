import pytest
import numpy as np

import pandas as pd
from sklearn.utils.testing import assert_array_equal

from sklearn_predict.feature_engineer import CountVectorizerSQL
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer

from skrules import SkopeRules
from sklearn.datasets import make_classification


def test_countvec():
    enc = CountVectorizer(binary=True)
    X = [
        "This is the first document.",
        "This is the second second document.",
        "And the third one.",
        "Is this the first document?",
    ]
    enc.fit(X)

    engine = create_engine("sqlite://", echo=False)
    data = pd.DataFrame(X, columns=["text"])
    data.to_sql("tbl", con=engine)

    text_cleaner = CountVectorizerSQL(enc).text_cleaner(text_column="text")
    sql_query = "SELECT {} as text from tbl".format(text_cleaner)
    data_clean = pd.read_sql(sql_query, engine)
    data_clean.to_sql("text_tbl", con=engine)

    queries = CountVectorizerSQL(enc).export(prefix="text")

    cv = ",".join(queries)
    sql_query = "SELECT {}, {} from text_tbl".format("text", cv)
    data_out = pd.read_sql(sql_query, engine).sum()
    assert data_out["text_and"] > 0
    assert data_out["text_document"] > 0
    assert data_out["text_first"] > 0
    assert data_out["text_is"] > 0
    assert data_out["text_second"] > 0
    assert data_out["text_the"] > 0
