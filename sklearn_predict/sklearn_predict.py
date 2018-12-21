"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import warnings

def sklearn_sql(model, feature_names):
    """
    Takes in a model and feature names to build classifier
    ```
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification()
    model = SGDClassifier(max_iter = 1000, tol=1e-3)
    model.fit(X, y)
    sklearn_sql(model, ["c{}".format(idx) for idx in range(20)])
    ```
    """
    if len(model.coef_.shape) == 1:
        query = ' + '.join(["(({}) * (`{}`))".format(coef, nm) for coef, nm in zip(model.coef_, feature_names)])
        query += " + {}".format(model.intercept_)
        query_list = [query]
    else:
        query_list = []
        n_classes = model.coef_.shape[0]
        for n_class in range(n_classes):
            query = ' + '.join(["(({}) * (`{}`))".format(coef, nm) for coef, nm in zip(model.coef_[n_class,:], feature_names)])
            query += " + {}".format(model.intercept_)
            query_list.append(query)
    return query_list


def linear_model(x, model, feature_names=None):
    n_features = model.coef_.shape[1]
    scores = np.dot(x, model.coef_.T) + model.intercept_
    # then we take argmax
    if scores.shape[1]==1:
        indices = (scores > 0).astype(np.int)
    else:
        indices = scores.argmax(axis=1)
    return model.classes_[indices]


