"""
This module converts linear models to the SQL counterparts. 

As the target language is sqlite, this may mean some models such as logistic regression
is out of reach. Nevertheless, it may be implemented here for completeness.

TODO: logistic regression via softmax (logistic function) and normalising...
"""

import numpy as np
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
