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

    def export(self, config={}):
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

