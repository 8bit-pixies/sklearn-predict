.. -*- mode: rst -*-

sklearn-predict - Run predictions inside the database
=====================================================

.. _scikit-learn: https://scikit-learn.org
.. _tidypredict: https://db.rstudio.com/tidypredict/

**sklearn-predict** Run predictions inside the database.

It aims to transcompile scikit-learn_ models to SQL equivalents in a
similar manner to tidypredict_.

Motivation
==========

.. _sklearn-porter: https://github.com/nok/sklearn-porter

Often when we're building Machine Learning models, the challenges are around
how can these models be built and deployed? How can we make these models smarter?

In many systems the complexity of the algorithms are often limited - this could
be due to the languages or libraries which are used, the latency or complexity
requirements which are in place, or other contraints. 

In this package, we aim to provide a _minimal_ set of tools to enable
machine learning deployment across wide variety of contexts. Whereas packages like
sklearn-porter_ aim to support the final classifier, we aim to provide opinionated
feature engineering tools to support more complex workflows.

Goals
=====

To provide a pathway for Python users to deploy onto databases; through enabling simple
rule-based features and linear models

Non-Goals
=========

We do not aim to employ a range of complex mathematical transformations in our implementation.
For testing, we will leverage sqlite database as the limit of the allowable in-built 
transformations for our machine learning pipelines. 

The reasoning is that many more complex systems which we may aim to embed will likewise
only support a small subset of functions. We believe that these systems too should
be enabled to deployed more complex machine learning workflows in spite of such limitations

Testing
=======

.. code:: bash

    pytest --capture=sys