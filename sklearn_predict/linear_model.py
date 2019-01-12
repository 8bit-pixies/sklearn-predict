"""
This module converts linear models to the SQL counterparts. 

As the target language is sqlite, this may mean some models such as logistic regression
is out of reach. Nevertheless, it may be implemented here for completeness.

TODO: logistic regression via softmax (logistic function) and normalising...
"""

import numpy as np
from copy import deepcopy
import warnings

class LinearSVMSQL:
    def __init__(self, model, feature_names, language='sql'):
        self.model = model
        self.language = language
        self.feature_names = feature_names

    def export(self):
        # TODO: add checks that attr exists via hasattr
        if len(self.model.coef_.shape) == 1:
            query = ' + '.join(["(({}) * (`{}`))".format(coef, nm) for coef, nm in zip(self.model.coef_, self.feature_names)])
            query += " + {}".format(self.model.intercept_)
            query_list = [query]
        else:
            query_list = []
            n_classes = self.model.coef_.shape[0]
            for n_class in range(n_classes):
                query = ' + '.join(["(({}) * (`{}`))".format(coef, nm) for coef, nm in zip(self.model.coef_[n_class,:], self.feature_names)])
                query += " + {}".format(self.model.intercept_[n_class])
                query_list.append(query)
        return query_list

    def to_sql(self, tbl_name, pred_prefix="pred", pred_target=None, pred_name="prediction"):
        """
        tbl_name: name of the table in the database
        pred_prefix: the name of the prefix of each pred item
        pred_target: the categorical names for the prediction if provided
        pred_name: the name of the final prediction - only used for multiclass prediction

        Examples

        To generate query with output column name as "alert" we would use:
        ```
        LinearSVMSQL(est, feature_names).to_sql(tbl_name="tbl", pred_prefix="", pred_target=["alert"])
        ```
        """
        def get_max(pred_list, pred_names, prediction_name):
            #assert len(pred_list) == pred_target
            query_builder = "CASE"
            for pred, nm in zip(pred_list, pred_names):
                str_builder = []
                for pred_ in pred_list:
                    if pred == pred_:
                        continue
                    str_builder.append("{} > {}".format(pred, pred_))
                str_builder_ = ' AND '.join(str_builder)
                str_builder_ = " WHEN {} THEN {} ".format(str_builder_, nm)
                query_builder += str_builder_
            query_builder += "END AS {}".format(prediction_name)
            return query_builder

        queries = self.export()
        if pred_target is None:
            pred_target = range(len(queries))

        # fix this pattern...
        if pred_prefix != "":
            pred_names = ["{}_{}".format(pred_prefix, nm) for nm in pred_target]
        else:
            pred_names = pred_target
        
        select_query = ','.join(["{} > 0 as {}".format(x, idx) for idx, x in zip(pred_names, queries)])
        
        if len(queries) == 1:
            # we only have one single prediction column
            sql_query = "SELECT {} from {}".format(select_query, tbl_name)
            return sql_query
        else:
            sub_query = "(SELECT {} from {})".format(select_query, tbl_name)
            case_query = get_max(pred_names, pred_target, pred_name)
            sql_query = "SELECT\n\t*,\n\t{}\nfrom {}".format(case_query, sub_query)
            return sql_query


class GraftingLinearModel:
    def __init__(self, model):
        # TODO: add checks for supported model types
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def base_size(self):
        # check_is_fitted(self.model.coef_)
        hasattr(self.model, 'coef_')
        n_features = self.model.coef_.shape[1]
        return n_features
    
    def mask_model(self, top_k=None, alpha=None):
        """
        Mask model filters coefficients by norm (top_k or by alpha)
        and returns the coef matrix and model.

        It does not update the underlying model object
        """
        if top_k is None and alpha is None:
            return self.model, self.model.coef_, self.model.intercept_
        
        model = deepcopy(self.model)
        new_coef = model.coef_.copy()
        
        coef_norm = np.linalg.norm(new_coef, ord=2, axis=0)
        #coef_sort = np.argsort(coef_norm)[::-1]
        if top_k is not None:
            coef_filter = np.argsort(coef_norm)[-top_k:][::-1]
        elif alpha is not None:
            coef_filter = coef_norm >= alpha
        else:
            raise Exception("Could not get new model with top_k: {}, alpha: {}".format(top_k, alpha))
        new_coef = new_coef[:, coef_filter]
        model.coef_ = new_coef
        new_intercept = model.intercept_.copy() # intercepts are class-wise
        return model, new_coef, new_intercept
    
    def graft_coef(self, max_features=None):
        n_features = self.base_size()
        if max_features is None or max_features == n_features:
            return self
        
        if max_features < n_features:
            raise Exception("Grafting size is less dimensions than previously fitted!")
        
        n_classes = self.model.coef_.shape[0]
        new_coef = np.zeros(shape=(n_classes, max_features))
        new_coef[:, :n_features] = self.model.coef_
        self.model.coef_ = new_coef.copy()

        # new_intercept = self.model.intercept_ # need checking?
        # self.model.intercept_ = new_intercept.copy()
        return self

    def partial_fit(self, X, y):
        n_features = X.shape[1]
        self.graft_coef(n_features)
        self.model.partial_fit(X, y)
        return self





