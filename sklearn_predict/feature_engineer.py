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


class OneHotEncoderSQL:
    """
    Usage:

    ```
    from sklearn.preprocessing import OneHotEncoder
    
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)

    export = OneHotEncodingSQL(enc, ["gender", "class"])
    #print(export.export())
    ```
    """
    def __init__(self, model, column_names, feature_names=None):
        self.model = model
        self.column_names = column_names
        self.feature_names = (
            feature_names
            if feature_names is not None
            else self.get_feature_names(model, column_names)
        )

    def get_feature_names(self, model, column_names):
        feature_names = []
        for col_names, fnames in zip(column_names, model.categories_):
            feature_names.extend(
                ["{}_{}".format(col_names, x) for x in fnames.tolist()]
            )
        return feature_names

    def export(self):
        """
        One hot encoding has all the feature names
        and all the transformation information required.
        """
        queries = []
        for col_names, fnames in zip(self.column_names, self.model.categories_):
            for feature_name in fnames:
                queries.append(
                    'CASE WHEN `{col}` = "{feat}" then 1 else 0 end as `{col}_{feat}`'.format(
                        col=col_names, feat=feature_name
                    )
                )
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
                cond = re.sub(find_cond, r" `\g<1>` ", cond)
            return cond

        binary_template = "CASE WHEN {} THEN 1 ELSE 0"
        return [fix_names(binary_template.format(x[0])) for x in self.model.rules_]


class CountVectorizerSQL:
    def __init__(self, model):
        self.model = model
        if not self.model.binary:
            warnings.warn("CountVectorizerSQL only formally supports `binary`=True")

    def text_cleaner(self, text_column="text", config={"concat_func": False}):
        # TODO: what options should be encoded?
        word_boundaries = list("!_-;?,(){}=+ ")
        if config.get("concat_func", True):
            select_col = "CONCAT(CONCAT('.', LOWER(`{}`)), '.')".format(text_column)
        else:
            select_col = "('.' || LOWER(`{}`) || '.')".format(text_column)
        for punc in word_boundaries:
            select_col = "REPLACE(" + select_col + ", '{}', '.')".format(punc)
        return select_col

    def export(self, text_column="text", prefix="", mode="sqlite"):
        """
        Model only works for binary (1, 0) encoding, instead of count vectorizer.
        This is design for lightweight text mining
        """
        feature_names = self.model.get_feature_names()
        feature_query = []
        for name in feature_names:
            if prefix != "":
                prefix_name = "{}_{}".format(prefix, name)
            else:
                prefix_name = name
            query = "CASE WHEN {text} LIKE '%.{name}.%' THEN 1 ELSE 0 END AS `{prefix_name}`".format(
                text=text_column, name=name, prefix_name=prefix_name
            )
            # if mode != "ksql":
            #    query = "CASE WHEN INSTR({text}, '.{name}.') > 0 THEN 1 ELSE 0 END AS {prefix_name}".format(text=text_column, name=name, prefix_name=prefix_name)
            # else:
            #    query = "CASE WHEN {text} LIKE '%.{name}.%' THEN 1 ELSE 0 END AS {prefix_name}".format(text=text_column, name=name, prefix_name=prefix_name)
            feature_query.append(query)
        return feature_query


class BinaryEncoding:
    """
    Reimplementationg of binary encoding. The algorithm works in two stages:

    1.  Converts categorical data to ordinal encoding
    2.  Converts ordinal encoding to base 2

    The base 2 representation is used as the final representation of the data.
    This is essentially one-hot but reduced by an order of 2

    ```py
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder()
    X = [["Male", 1], ["Female", 3], ["Female", 2]]
    enc.fit(X)
    enc.categories_

    binary_encode = BinaryEncoding(enc, ["gender", "age"])
    binary_encode.export()

    X = [[x] for x in range(10)]
    enc.fit(X)
    enc.categories_

    binary_encode = BinaryEncoding(enc, ["x"])
    binary_encode.export()
    ```
    """

    def __init__(self, model, column_names):
        self.model = model
        self.column_names = column_names

    def num2binstr(self, num, pad=0):
        return bin(num)[2:].rjust(pad, "0")

    def export(self):
        """
        generate the binary encoding information        
        """
        queries = []
        for col_names, fnames in zip(self.column_names, self.model.categories_):
            query = {}
            num_cols = len(self.num2binstr(len(fnames)))
            # have to split the categories to the modulus
            for ord_indx in range(num_cols):
                # ord is 0, 1, 2, 3
                new_col = f"{col_names}_{ord_indx}"
                for sql_column, feat in enumerate(fnames):
                    bin_sql_column = self.num2binstr(sql_column, pad=num_cols)
                    print(feat, sql_column, bin_sql_column)
                    print("\t", "Checking...index", ord_indx)
                    if bin_sql_column[ord_indx] == "1":
                        query_update = query.get(new_col, [])
                        query_update.append(f'`{col_names}` = "{feat}"')
                        query[new_col] = query_update.copy()
            queries.append(query)

        # reformat all queries here
        sql_query = []
        template = "CASE WHEN {cond} THEN 1 ELSE 0 END AS {key}"
        for query in queries:
            for k in query.keys():
                sql_query.append(template.format(key=k, cond=" OR ".join(query[k])))
        return sql_query
