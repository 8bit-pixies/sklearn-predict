"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import warnings

def linear_svc(model, feature_names):
    """
    Takes in a model and feature names to build classifier
    ```
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification()
    model = SGDClassifier(max_iter = 1000, tol=1e-3)
    model.fit(X, y)
    linear_svc(model, ["c{}".format(idx) for idx in range(20)])
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
            query += " + {}".format(model.intercept_[n_class])
            query_list.append(query)
    return query_list


def linear_logistic(model, feature_names):
    """
    Takes in a model and feature names to build classifier
    ```
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification()
    model = SGDClassifier(max_iter = 1000, tol=1e-3, loss='log')
    model.fit(X, y)
    linear_logistic(model, ["c{}".format(idx) for idx in range(20)])
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
            query += " + {}".format(model.intercept_[n_class])
            # logistic...
            query = ""
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



def linearsvm_sql(model, feature_names):
    """
    Takes in a model and feature names to build classifier
    ```
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification()
    model = SGDClassifier(max_iter = 1000, tol=1e-3)
    model.fit(X, y)
    linearsvm_sql(model, ["c{}".format(idx) for idx in range(20)])
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
            query += " + {}".format(model.intercept_[n_class])
            query_list.append(query)
    return query_list


def skrules_sql(model, feature_names):
    """
    Takes in a skopt model to build new features in database
    ```
    from skrules import SkopeRules
    from sklearn.datasets import make_classification
    X, y = make_classification()
    feature_names = ["c{}".format(idx) for idx in range(20)]
    clf = SkopeRules(max_depth_duplication=2,
                 n_estimators=30,
                 precision_min=0.3,
                 recall_min=0.1,
                 feature_names=feature_names)
    clf.fit(X, y)
    print(clf.rules_)
    skrules_sql(clf, feature_names)
    ```
    """
    def fix_names(cond):
        import re
        for nm in feature_names:
            find_cond = " ({}) ".format(nm)
            cond = re.sub(find_cond, r' `\g<1>` ', cond)
        return cond

    binary_template = "CASE WHEN {} THEN 1 ELSE 0"
    return [fix_names(binary_template.format(x[0])) for x in model.rules_]