"""
This converts parts of the feature engineering to sql code

In this module, we will cover parts of:
*  sklearn.preprocessing
*  sklearn.feature_extraction

In the future if this actually becomes more mature, we may think about
splitting up the modules. For now it will be encapsulated into 
one module only

"""


import numpy as np
import warnings

class OneHotEncodingSQL:
    def __init__(self, model, column_names, feature_names=None):
        self.model = model
        self.column_names = column_names
        self.feature_names = feature_names if feature_names is not None else self.get_feature_names(model, column_names)
    
    def get_feature_names(self, model, column_names):
        feature_names = []
        for col_names, fnames in zip(column_names, model.categories_):
            feature_names.extend(["{}_{}".format(col_names, x) for x in fnames.tolist()])
        return feature_names

    def export(self):
        """
        One hot encoding has all the feature names
        and all the transformation information required.
        """
        queries = []
        for col_names, fnames in zip(self.column_names, self.model.categories_):
            for feature_name in fnames:
                queries.append("CASE WHEN `{col}` = \"{feat}\" then 1 else 0 end as `{col}_{feat}`".format(col=col_names, feat=feature_name))
        return queries


class SkopeRulesSQL:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def export(self):
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
            for nm in self.feature_names:
                find_cond = " ({}) ".format(nm)
                cond = re.sub(find_cond, r' `\g<1>` ', cond)
            return cond

        binary_template = "CASE WHEN {} THEN 1 ELSE 0"
        return [fix_names(binary_template.format(x[0])) for x in self.model.rules_]

def skrules_sql(model, feature_names):

    def fix_names(cond):
        import re
        for nm in feature_names:
            find_cond = " ({}) ".format(nm)
            cond = re.sub(find_cond, r' `\g<1>` ', cond)
        return cond

    binary_template = "CASE WHEN {} THEN 1 ELSE 0"
    return [fix_names(binary_template.format(x[0])) for x in model.rules_]